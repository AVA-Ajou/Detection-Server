#!/usr/bin/env python3
"""베이스 모델 하나에 어댑터 여러 개를 얹고, 위험도와 근거를 뽑는다.

이 파일은 학습 저장소(`Voice-Detection`)의 코드를 **참조하지 않는다.** 프롬프트는
어댑터 폴더의 `prompt.json`에서 읽어 재구성한다. 코드를 복사해 쓰면 언젠가 갈라지고,
갈라지는 순간 에러 없이 확률만 조용히 틀어지기 때문이다.

어댑터 폴더가 갖춰야 할 것 — 학습 스크립트가 전부 함께 저장한다.

    adapter_model.safetensors   가중치
    adapter_config.json         LoRA 설정
    prompt.json                 지시문 / 입력 상한 / 정답 후보  ← 계약
    calibration.json            온도. 없으면 보정 없이 간다
    chat_template.jinja 등      토크나이저
"""

import json
import re
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
ADAPTERS = HERE / "adapters"

# 사기 진행 단계. 확률과는 **다른 값**이다.
#
# 확률은 "피싱이 맞나"이고 어댑터가 로짓으로 답한다. 단계는 "지금 어디까지 왔나"이고 원본
# Gemma가 전사본을 읽어 답한다. 같은 99%라도 사칭만 한 상태와 계좌번호를 부르는 상태는
# 사용자가 해야 할 행동이 다르다.
#
# **모델에게 확률을 묻지는 않는다.** 학습으로 점검된 적 없는 숫자가 나오고, 판정을 알려주면
# 그 전제에 끌려가 사실상 이진값이 된다. 반면 단계는 전사본에 적힌 사실을 읽는 일이라
# 모델이 잘하고, 사람이 전사본을 보고 채점할 수 있어 검증도 된다.
#
# 구분은 calibrate.py 의 신호 그룹(IMPERSONATION / PRETEXT / DEMAND)을 그대로 따른다.
STAGES = {
    1: "접촉·사칭",
    2: "압박·유인",
    3: "정보·계좌",
    4: "이체 지시",
}

# 판정 결과에 따라 묻는 말이 달라야 한다. "피싱으로 판정되었다"를 무조건 전제로 깔면
# 정상 통화에도 모델이 억지 근거를 만들어낸다 — 배송 지연 사과를 "감정에 호소하는 사기
# 수법"으로 둔갑시키는 것을 실제로 확인했다. 모델은 주어진 전제를 의심하지 않는다.
ASSESS_HIGH = """다음은 통화 전사본이다.

{text}

이 통화는 보이스피싱으로 판정되었다.
사기가 어느 단계까지 진행됐는지 고르고, 그렇게 본 근거를 전사본에서 **직접 인용**해 한두
문장으로 써라.

1 접촉·사칭 — 기관이나 사람을 사칭했지만 아직 요구가 없다
2 압박·유인 — 범죄 연루·구속으로 겁을 주거나, 대출·환급을 미끼로 건다
3 정보·계좌 — 개인정보나 계좌번호를 요구하거나 불러준다. 앱 설치·원격 유도
4 이체 지시 — 송금·현금 전달·수거책 방문을 실제로 지시한다

아직 나오지 않은 단계를 앞질러 고르지 마라. 전사본에 없는 내용은 지어내지 마라.

형식:
단계: <숫자 하나>
근거: <한두 문장>"""

EXPLAIN_LOW = """다음은 통화 전사본이다.

{text}

이 통화는 보이스피싱이 **아닌** 것으로 판정되었다(위험도 {risk}점).
왜 위험하지 않다고 볼 수 있는지 한두 문장으로 설명하라.
없는 위험 신호를 억지로 찾아내지 마라."""

# 이 값 위아래로 묻는 말이 갈린다. 아래면 단계를 묻지 않는다 — 정상 통화에 단계를 붙이면
# 그 자체가 "사기가 진행 중"이라는 잘못된 신호가 된다.
ASSESS_THRESHOLD = 50

# 형식을 지킨 답(`단계: 3`)과 풀어 쓴 답(`3단계로 보입니다`) 양쪽을 받는다.
# 앞의 것을 먼저 보는 이유는 "3단계"가 근거 문장 안에서 다른 뜻으로 쓰일 수 있어서다.
_STAGE_RES = (re.compile(r"단계\s*[:：]\s*([1-4])"),
              re.compile(r"단계\s*([1-4])"),
              re.compile(r"([1-4])\s*단계"))
_REASON_RE = re.compile(r"근거\s*[:：]\s*(.+)", re.S)
_MARKUP_RE = re.compile(r"\*+")


class Task:
    """어댑터 하나와 그 계약."""

    def __init__(self, name, path, tokenizer):
        self.name = name
        self.path = path
        self.tokenizer = tokenizer

        contract = json.loads((path / "prompt.json").read_text(encoding="utf-8"))
        self.instruction = contract["instruction"]
        self.max_text_tokens = contract["max_text_tokens"]
        self.base_model = contract.get("base_model")

        # 정답 후보의 첫 토큰. 확률은 이 한 자리에서만 읽는다.
        self.yes = tokenizer.encode(contract["yes"], add_special_tokens=False)[0]
        self.no = tokenizer.encode(contract["no"], add_special_tokens=False)[0]
        if self.yes == self.no:
            raise SystemExit(f"[{name}] 정답 후보의 첫 토큰이 같습니다.")

        cal = path / "calibration.json"
        self.temperature = json.loads(cal.read_text())["temperature"] if cal.exists() else 1.0
        self.calibrated = cal.exists()

    def prompt_ids(self, text):
        """학습 때와 **동일한** 프롬프트를 만든다. 순서가 한 군데라도 어긋나면 안 된다."""
        tok = self.tokenizer
        clipped = tok.decode(tok.encode(text, add_special_tokens=False)[:self.max_text_tokens])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": f"{clipped}\n\n{self.instruction}"}],
            add_generation_prompt=True, tokenize=False,
        )
        # 템플릿이 <bos>를 이미 넣으므로 특수 토큰을 또 붙이지 않는다.
        return tok.encode(rendered, add_special_tokens=False)


class Engine:
    """모델을 한 번만 올리고 재사용한다.

    어댑터를 태스크마다 따로 서빙하면 8GB짜리 베이스를 중복 적재하게 된다. LoRA는 원본을
    건드리지 않으므로 어댑터만 갈아끼우면 된다 — 통화·문자를 합쳐도 8.25GB로 끝난다.
    """

    def __init__(self, base_model=None, device=None):
        import torch

        self.torch = torch
        self.lock = threading.Lock()  # 순전파를 직렬화한다. GPU 하나에 동시 진입은 못 한다
        self.tasks = {}

        dirs = sorted(p for p in ADAPTERS.glob("*") if (p / "prompt.json").exists())
        if not dirs:
            raise SystemExit(f"{ADAPTERS} 아래에 어댑터가 없습니다. README를 참고하세요.")

        base_model = base_model or json.loads(
            (dirs[0] / "prompt.json").read_text(encoding="utf-8")).get("base_model")

        self.device, kwargs = self._placement(device)
        print(f"베이스 적재 — {base_model}  ({self.device})")
        self.model = self._open(base_model, kwargs)

        from peft import PeftModel
        from transformers import AutoTokenizer

        for i, path in enumerate(dirs):
            name = path.name
            if i == 0:
                self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name)
            else:
                self.model.load_adapter(str(path), adapter_name=name)
            # 토크나이저는 어댑터와 함께 저장된 것을 쓴다. 대화 템플릿까지 같아야 한다.
            self.tasks[name] = Task(name, path, AutoTokenizer.from_pretrained(path))
            print(f"  어댑터 '{name}'  온도 {self.tasks[name].temperature:.3f}"
                  f"{'' if self.tasks[name].calibrated else '  (보정 없음)'}")
        self.model.eval()

    def _placement(self, device):
        """어디에 어떻게 올릴지. 맥과 GPU 서버에서 답이 다르다."""
        torch = self.torch
        if device != "cpu" and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return "cuda", {
                "attn_implementation": "sdpa",
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute),
            }
        if device != "cpu" and torch.backends.mps.is_available():
            # bitsandbytes가 CUDA 전용이라 맥에서는 4bit를 못 쓴다. bf16 원본을 올린다 —
            # 24GB면 압축할 이유가 없고, 압축 푸는 비용이 사라져 오히려 빠르다.
            return "mps", {"attn_implementation": "sdpa", "dtype": torch.bfloat16,
                           "device_map": {"": "mps"}}
        return "cpu", {"attn_implementation": "sdpa", "dtype": torch.float32}

    def _open(self, model_id, kwargs):
        """베이스 모델을 연다. 승인이 필요한 저장소면 할 일을 알려주고 끝낸다.

        **자동으로 미러로 갈아타지 않는다.** 다른 가중치에 어댑터가 붙으면 에러 없이 확률만
        틀어진다. 어느 가중치를 쓸지는 사람이 정해야 하는 문제다.
        """
        try:
            return self._open_any(model_id, kwargs)
        except OSError as e:
            if "gated repo" not in str(e) and "401" not in str(e):
                raise
            raise SystemExit(
                f"\n[{model_id}] 는 승인이 필요한 저장소인데 인증 정보가 없습니다.\n"
                f"둘 중 하나를 하세요.\n\n"
                f"  ① 정품을 쓴다 — https://huggingface.co/{model_id} 에서 라이선스 동의 후\n"
                f"       huggingface-cli login\n\n"
                f"  ② 같은 가중치의 공개 미러를 쓴다 (승인 불필요)\n"
                f"       BASE_MODEL=unsloth/gemma-3-4b-it \\\n"
                f"           uvicorn app:app --host 0.0.0.0 --port 8000\n\n"
                f"  미러를 썼다면 알려진 값으로 반드시 대조하세요 (docs/api.md).\n"
                f"  기준선은 ../Voice-Detection/docs/RESULTS.md 에 있습니다.\n"
            ) from None

    def _open_any(self, model_id, kwargs):
        # Gemma 3는 크기에 따라 클래스가 갈린다 — 1B는 순수 언어 모델, 4B 이상은 멀티모달
        # 래퍼다. 어느 쪽이든 텍스트만 넣으면 같은 형태의 로짓이 나오므로 되는 쪽으로 연다.
        from transformers import AutoModelForCausalLM
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    # ────────────────────────────────────────── 추론

    def risk(self, task_name, text):
        """P("예"). 생성하지 않는다 — 순전파 한 번으로 끝난다."""
        torch = self.torch
        task = self.task(task_name)
        ids = torch.tensor([task.prompt_ids(text)]).to(self.model.device)

        with self.lock:
            self.model.set_adapter(task.name)
            with torch.no_grad():
                logits = self.model(input_ids=ids).logits[0, -1]

        pair = torch.tensor([logits[task.yes], logits[task.no]],
                            dtype=torch.float32) / task.temperature
        return torch.softmax(pair, dim=0)[0].item()

    def assess(self, task_name, text, risk):
        """어댑터를 뗀 원본 Gemma에게 **진행 단계와 근거**를 받는다.

        근거를 학습시키지 않은 이유 — 원문을 읽고 인용해 설명하는 건 대화형 모델이 이미
        잘하는 일이고, 규칙이 만든 문장을 학습시키면 그 규칙의 편협함까지 배운다.

        위험도가 낮으면 단계를 묻지 않는다. 정상 통화에 단계를 붙이면 그 자체가
        "사기가 진행 중"이라는 잘못된 신호가 된다.
        """
        torch = self.torch
        task = self.task(task_name)
        tok = task.tokenizer
        clipped = tok.decode(tok.encode(text, add_special_tokens=False)[:task.max_text_tokens])
        score = round(risk * 100, 1)
        staged = score >= ASSESS_THRESHOLD

        content = (ASSESS_HIGH if staged else EXPLAIN_LOW).format(text=clipped, risk=score)
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False,
        )
        ids = tok.encode(rendered, add_special_tokens=False)
        prompt = torch.tensor([ids]).to(self.model.device)

        with self.lock:
            with self.model.disable_adapter():  # 이 블록 안에서만 원본 Gemma로 돌아간다
                with torch.no_grad():
                    # 단계 하나와 두세 문장이면 충분하다. 256으로 두면 MPS에서 15초를 넘겨
                    # 앱의 타임아웃에 걸린다.
                    out = self.model.generate(prompt, max_new_tokens=160, do_sample=False)
        raw = tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()
        return self._parse(raw, staged)

    @staticmethod
    def _parse(raw, staged):
        """`단계: 3 / 근거: …` 를 뜯는다. **형식을 어길 것을 전제로 짠다.**

        4B 모델이라 형식을 지키지 못하는 경우가 있다. 그때도 근거 문장은 살려야 하므로,
        단계를 못 읽으면 None으로 두되 원문을 그대로 근거로 쓴다 — 판정 자체는 어댑터가
        이미 내놨고 이건 부가 정보다.
        """
        if not staged:
            return {"stage": None, "stage_label": None, "reason": raw}

        stage = next((int(m.group(1)) for m in
                      (r.search(raw) for r in _STAGE_RES) if m), None)

        reason = _REASON_RE.search(raw)
        reason = reason.group(1) if reason else raw
        # 모델이 붙이는 마크다운 강조(`**근거:**`)의 잔재를 걷어낸다. 앱은 서식 없는
        # 한 줄로 표시하므로 별표가 그대로 화면에 나온다.
        reason = _MARKUP_RE.sub("", reason).strip()
        return {
            "stage": stage,
            "stage_label": STAGES.get(stage),
            "reason": reason,
        }

    def analyze(self, task_name, text, want_reason=False):
        started = time.perf_counter()
        p = self.risk(task_name, text)
        result = {"task": task_name, "risk": round(p * 100, 1), "probability": p}
        if want_reason:
            result.update(self.assess(task_name, text, p))
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return result

    def task(self, name):
        if name not in self.tasks:
            raise KeyError(f"'{name}' 어댑터가 없습니다. 가능한 값: {sorted(self.tasks)}")
        return self.tasks[name]

    def status(self):
        return {
            "device": self.device,
            "tasks": {n: {"temperature": round(t.temperature, 3),
                          "calibrated": t.calibrated,
                          "max_text_tokens": t.max_text_tokens,
                          "base_model": t.base_model} for n, t in self.tasks.items()},
        }

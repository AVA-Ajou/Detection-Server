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

from . import signals

# 이 파일은 src/ 안에 있고 어댑터는 저장소 루트에 있다.
ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"

# 판정 결과에 따라 묻는 말이 달라야 한다. "피싱으로 판정되었다"를 무조건 전제로 깔면
# 정상 통화에도 모델이 억지 근거를 만들어낸다 — 배송 지연 사과를 "감정에 호소하는 사기
# 수법"으로 둔갑시키는 것을 실제로 확인했다. 모델은 주어진 전제를 의심하지 않는다.
#
# **단계는 더 이상 여기서 묻지 않는다.** 전에는 이 프롬프트가 근거와 단계 숫자를 함께
# 받았는데, 인용은 정확한 반면 숫자는 한쪽으로 쏠렸다. 정답지 36건 실측 — 생성 33.3%,
# 규칙 91.7%. 숫자는 `signals.stage_of()` 가 주고, 모델은 잘하는 일(인용)만 한다.
EXPLAIN_HIGH = """다음은 통화 전사본이다.

{text}

이 통화는 보이스피싱으로 판정되었다.
그렇게 볼 수 있는 근거를 전사본에서 **직접 인용**해 한두 문장으로 써라.
전사본에 없는 내용은 지어내지 마라."""

EXPLAIN_LOW = """다음은 통화 전사본이다.

{text}

이 통화는 보이스피싱이 **아닌** 것으로 판정되었다(위험도 {risk}점).
왜 위험하지 않다고 볼 수 있는지 한두 문장으로 설명하라.
없는 위험 신호를 억지로 찾아내지 마라."""

# 이 값 위아래로 묻는 말이 갈린다. 아래면 단계를 매기지 않는다 — 정상 통화에 단계를 붙이면
# 그 자체가 "사기가 진행 중"이라는 잘못된 신호가 된다.
#
# 앱(`BackendClassificationClient.signalOf`)의 `CAUTION_THRESHOLD` 와 **같은 값이어야 한다.**
# 어긋나면 앱은 등급을 올리는데 서버는 단계를 안 주는(또는 그 반대) 상태가 되고, 단계는
# 등급을 가르는 두 재료 중 하나라서 그대로 오등급이 된다.
#
# 70 → 50 → 33 → 20 으로 내려왔다. 지금 이 값은 "피싱이라고 부르는 선"이 아니라 **단계를
# 계산해 볼 만한 최소선**이다. 앱이 세 문턱으로 판정하기 때문이다.
#
#   위험도 80 이상          앱이 경보(빨강). 단계를 보지 않는다
#   위험도 50 이상 80 미만   단계가 있으면 `경보`(빨강), 없으면 `경보우려`(주황)
#   위험도 20 이상 50 미만   단계가 있으면 주의보(주황), 없으면 예보(노랑)
#
# 그래서 20 아래로는 계산할 이유가 없고, 20 위로는 **앱이 판단할 재료를 서버가 쥐고 있으면
# 안 된다.** 이 값을 앱의 `CAUTION_THRESHOLD` 보다 높이면 그 사이 구간이 단계를 못 받아
# 통째로 예보로 내려앉는다 — 주의보가 영영 나오지 않는다.
ASSESS_THRESHOLD = 20

# 전사 지시문. 요약·교정을 막는 것이 요점이다 — 대화형 모델이라 시키지 않으면 말을 다듬으려
# 하는데, 판정은 원문의 말버릇과 표현을 봐야 하므로 들린 그대로여야 한다.
TRANSCRIBE_INSTRUCTION = "이 음성을 한국어로 그대로 받아적어라. 요약하거나 고치지 마라."

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

        # 오디오를 받는 베이스라면 전사도 이 모델이 한다. 프로세서(오디오 전처리기)가 있어야
        # 하므로 여기서 한 번만 준비하고, 안 되면 None 으로 두어 /transcribe 가 거절하게 한다.
        self.processor = self._load_processor(base_model)
        print(f"오디오 입력  {'가능' if self.processor else '불가 — 전사는 이 모델로 못 한다'}")

    def _load_processor(self, model_id):
        from transformers import AutoConfig, AutoProcessor
        try:
            if not hasattr(AutoConfig.from_pretrained(model_id), "audio_config"):
                return None
            return AutoProcessor.from_pretrained(model_id)
        except Exception as e:  # 전처리기 의존성(pillow·torchvision 등)이 빠진 경우
            print(f"  프로세서 적재 실패 — 전사 없이 진행: {type(e).__name__}")
            return None

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
                f"       BASE_MODEL=unsloth/{model_id.split('/')[-1]} \\\n"
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

    def transcribe(self, task_name, audio_path):
        """음성을 글로 옮긴다. **어댑터를 끈 원본 모델**이 한다.

        베이스가 오디오를 받는 모델일 때만 된다(`config.json`에 `audio_config`가 있어야 한다).
        Gemma 3 에는 없고 Gemma 4 에는 있다 — 그래서 전에는 이 일을 외부 API(Groq Whisper)에
        맡겨야 했다.

        어댑터는 텍스트 판정용으로 학습돼 있어 전사에 쓰면 안 된다. 켠 채로 음성을 넣으면
        학습 분포 밖이라 결과를 신뢰할 수 없다.
        """
        torch = self.torch
        task = self.task(task_name)
        if self.processor is None:
            raise RuntimeError(
                "이 베이스 모델은 오디오를 받지 못합니다. Gemma 4 처럼 audio_config 가 있는 "
                "모델이어야 합니다."
            )

        messages = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": TRANSCRIBE_INSTRUCTION},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        with self.lock:
            with self.model.disable_adapter():
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )[0].strip()

    def stage(self, text, risk):
        """진행 단계. **정규식이라 비용이 없다** — 순전파도 생성도 하지 않는다.

        모델에게 묻지 않는 이유 — 4택 분류를 자유 생성으로 시키는 셈이라 한쪽으로 쏠린다.
        정답지 36건에서 생성 33.3% / 규칙 91.7% 였다(`stage_eval.py`).

        위험도가 낮으면 단계를 매기지 않는다. 정상 통화에 단계를 붙이면 그 자체가
        "사기가 진행 중"이라는 잘못된 신호가 된다.

        **규칙에는 자르지 않은 원문을 준다.** 정규식은 길이 비용이 없고, 이체 지시는 통화
        끝에 오는 일이 많아 1024토큰에서 잘리면 단계를 통째로 놓친다.
        """
        if round(risk * 100, 1) < ASSESS_THRESHOLD:
            return {"stage": None, "stage_label": None, "stage_evidence": []}
        stage, evidence = signals.stage_of(text)
        return {
            "stage": stage,
            "stage_label": signals.label(stage),
            "stage_evidence": [q for _, q in evidence],
        }

    def explain(self, task_name, text, risk):
        """판정 근거를 원본 모델이 원문을 인용해 쓴다. **160토큰 생성이라 십수 초 걸린다.**

        근거를 학습시키지 않은 이유 — 원문을 읽고 인용해 설명하는 건 대화형 모델이 이미
        잘하는 일이고, 규칙이 만든 문장을 학습시키면 그 규칙의 편협함까지 배운다.

        **앱은 이 값을 쓰지 않는다.** 인용 자체는 정확한데 가장 결정적인 문구를 못 고른다 —
        계좌번호를 부르는 대목 대신 "통화가 녹취됩니다"를 뽑아오는 것을 두 번 확인했다.
        화면에 나가는 근거는 `stage_evidence`(규칙이 뽑은 인용구)다. 이 함수는 진단용으로
        남겨둔다 — `reason: true` 를 명시할 때만 돈다.
        """
        torch = self.torch
        task = self.task(task_name)
        tok = task.tokenizer
        clipped = tok.decode(tok.encode(text, add_special_tokens=False)[:task.max_text_tokens])
        score = round(risk * 100, 1)
        # 판정에 따라 묻는 말이 갈린다. 정상 통화에 "피싱으로 판정되었다"를 전제로 깔면
        # 모델이 억지 근거를 만든다.
        high = score >= ASSESS_THRESHOLD

        content = (EXPLAIN_HIGH if high else EXPLAIN_LOW).format(text=clipped, risk=score)
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False,
        )
        ids = tok.encode(rendered, add_special_tokens=False)
        prompt = torch.tensor([ids]).to(self.model.device)

        with self.lock:
            with self.model.disable_adapter():  # 이 블록 안에서만 원본 Gemma로 돌아간다
                with torch.no_grad():
                    # 두세 문장이면 충분하다. 256으로 두면 MPS에서 15초를 넘겨 앱의
                    # 타임아웃에 걸린다.
                    out = self.model.generate(prompt, max_new_tokens=160, do_sample=False)
        raw = tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()
        return self._clean(raw)

    @staticmethod
    def _clean(raw):
        """근거 문장만 남긴다. **형식을 어길 것을 전제로 짠다.**

        작은 모델이라 형식을 지키지 못하는 경우가 있다. `근거:` 가 없으면 생성한 것을
        통째로 쓴다 — 판정 자체는 어댑터가 이미 내놨고 이건 부가 정보다.
        """
        m = _REASON_RE.search(raw)
        reason = m.group(1) if m else raw
        # 모델이 붙이는 마크다운 강조(`**근거:**`)의 잔재를 걷어낸다. 앱은 서식 없는
        # 한 줄로 표시하므로 별표가 그대로 화면에 나온다.
        return _MARKUP_RE.sub("", reason).strip()

    def analyze(self, task_name, text, want_reason=False):
        """위험도와 단계를 **한 번에** 준다.

        예전에는 단계를 받으려면 `reason: true` 를 켜야 했고, 그러면 근거 생성 때문에
        판정이 16초가 걸렸다. 단계는 정규식이라 공짜인데 비싼 것에 딸려 있던 셈이다.
        지금은 항상 함께 나가고, 근거 생성은 명시적으로 요청할 때만 돈다.
        """
        started = time.perf_counter()
        p = self.risk(task_name, text)
        result = {"task": task_name, "risk": round(p * 100, 1), "probability": p}
        result.update(self.stage(text, p))
        if want_reason:
            result["reason"] = self.explain(task_name, text, p)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return result

    def task(self, name):
        if name not in self.tasks:
            raise KeyError(f"'{name}' 어댑터가 없습니다. 가능한 값: {sorted(self.tasks)}")
        return self.tasks[name]

    def status(self):
        return {
            "device": self.device,
            "audio": self.processor is not None,
            "tasks": {n: {"temperature": round(t.temperature, 3),
                          "calibrated": t.calibrated,
                          "max_text_tokens": t.max_text_tokens,
                          "base_model": t.base_model} for n, t in self.tasks.items()},
        }

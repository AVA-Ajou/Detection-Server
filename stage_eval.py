#!/usr/bin/env python3
"""손으로 채점한 정답지에 규칙과 생성 모델을 나란히 걸어 단계 정확도를 잰다.

**이 파일이 있는 이유** — 규칙으로 바꾸는 게 실제로 나은지 숫자로 확인하지 않으면
"나아 보인다"에서 멈춘다. 오탐률 0.0% 가 개선을 못 재고 있는 것과 같은 함정이다.

정답지는 규칙을 만들기 **전에** 확정했다(`../Voice-Detection/eval/stage_gold.jsonl`).
규칙을 만든 쪽이 채점까지 하면 규칙이 맞히는 쪽으로 정답이 휘기 때문이다.

    python3 stage_eval.py              # 규칙만. 모델을 안 올려서 몇 초면 끝난다
    python3 stage_eval.py --llm        # 생성 모델 기준선까지 함께 (MPS에서 수 분)

`--llm` 은 **같은 3단계 정의**를 프롬프트에 주고 숫자만 뽑게 한다. 지금 서버가 쓰는 4단계
프롬프트와 비교하면 정의가 달라 불공평해지므로, 생성 모델에게 가장 유리한 조건
(정의 동일 + 근거 없이 숫자만)으로 맞춰준다. 그래도 규칙이 이기면 근거가 분명해진다.
"""

import argparse
import json
from pathlib import Path

import signals

HERE = Path(__file__).parent
GOLD = HERE.parent / "Voice-Detection" / "eval" / "stage_gold.jsonl"
TRANSCRIPTS = HERE.parent / "Voice-Detection" / "repo" / "Multimodal" / "data" / "transcripts"

LLM_PROMPT = """다음은 보이스피싱 통화 전사본이다.

{text}

사기가 어느 단계까지 진행됐는지 숫자 하나로만 답하라.

1 압박·유인 — 겁을 주거나 미끼를 걸었지만 아직 요구가 없다
2 정보·계좌 — 개인정보·계좌·앱 설치를 요구한다. 돈은 아직 움직이지 않았다
3 이체 지시 — 송금·입금·현금 전달을 실제로 지시한다

아직 나오지 않은 단계를 앞질러 고르지 마라. 숫자 하나만 쓰고 다른 말은 하지 마라."""


def load_gold(path):
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        src = TRANSCRIPTS / "vishing" / f"{row['id']}.json"
        row["text"] = json.loads(src.read_text(encoding="utf-8")).get("text", "")
        rows.append(row)
    return rows


def score(name, rows, key):
    """정확도와 ±1 이내. 단계를 못 낸 건(None)은 오답으로 센다."""
    exact = sum(1 for r in rows if r[key] == r["stage"])
    near = sum(1 for r in rows if r[key] is not None and abs(r[key] - r["stage"]) <= 1)
    blank = sum(1 for r in rows if r[key] is None)
    n = len(rows)
    print(f"  {name:10} 정확 {exact:3}/{n} ({exact/n*100:5.1f}%)   "
          f"±1 이내 {near:3}/{n} ({near/n*100:5.1f}%)   판단 못 함 {blank}건")
    return exact


def confusion(name, rows, key):
    print(f"\n{name} — 행이 정답, 열이 예측")
    print(f"{'':8}{'1':>6}{'2':>6}{'3':>6}{'없음':>6}")
    for gold in (1, 2, 3):
        cells = []
        for pred in (1, 2, 3, None):
            cells.append(sum(1 for r in rows if r["stage"] == gold and r[key] == pred))
        print(f"  정답 {gold}{cells[0]:6}{cells[1]:6}{cells[2]:6}{cells[3]:6}")


def run_llm(rows, base_model, device):
    """어댑터를 끈 원본 모델에게 단계 숫자만 받는다."""
    import re
    import torch
    from engine import Engine

    engine = Engine(base_model=base_model, device=device)
    task = engine.task("voice")
    tok = task.tokenizer
    number = re.compile(r"[1-3]")

    for i, row in enumerate(rows, 1):
        clipped = tok.decode(tok.encode(row["text"], add_special_tokens=False)[:task.max_text_tokens])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": LLM_PROMPT.format(text=clipped)}],
            add_generation_prompt=True, tokenize=False,
        )
        ids = tok.encode(rendered, add_special_tokens=False)
        prompt = torch.tensor([ids]).to(engine.model.device)
        with engine.lock:
            with engine.model.disable_adapter():
                with torch.no_grad():
                    out = engine.model.generate(prompt, max_new_tokens=8, do_sample=False)
        raw = tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()
        m = number.search(raw)
        row["llm"] = int(m.group()) if m else None
        print(f"  {i:3}/{len(rows)}  {row['id']:14} 정답 {row['stage']} → 모델 {row['llm']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--llm", action="store_true", help="생성 모델 기준선도 잰다")
    parser.add_argument("--base-model")
    parser.add_argument("--device")
    parser.add_argument("--show-misses", action="store_true", help="규칙이 틀린 건을 자세히 본다")
    args = parser.parse_args()

    rows = load_gold(args.gold)
    for row in rows:
        row["rule"], row["evidence"] = signals.stage_of(row["text"])

    print(f"정답지 {len(rows)}건 — "
          + " / ".join(f"{s}단계 {sum(1 for r in rows if r['stage'] == s)}" for s in (1, 2, 3)))

    if args.llm:
        print("\n생성 모델 기준선 측정 중")
        run_llm(rows, args.base_model, args.device)

    print("\n결과")
    score("규칙", rows, "rule")
    if args.llm:
        score("생성 모델", rows, "llm")

    confusion("규칙", rows, "rule")
    if args.llm:
        confusion("생성 모델", rows, "llm")

    if args.show_misses:
        print("\n규칙이 틀린 건")
        for r in rows:
            if r["rule"] == r["stage"]:
                continue
            print(f"\n  {r['id']}  정답 {r['stage']} → 규칙 {r['rule']}")
            print(f"    사람이 본 근거: {r['quote']}")
            for name, quote in r["evidence"]:
                print(f"    규칙이 본 근거: {name} — {quote}")

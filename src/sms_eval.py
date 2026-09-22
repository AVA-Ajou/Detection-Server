#!/usr/bin/env python3
"""문자 어댑터를 띄워둔 서버에 걸어 잰다 — 낱개 40건과 다채널 시나리오 7건.

`eval_hard_normal.py` 와 같은 자리다. 문자 어댑터는 무엇으로 학습됐는지 기록이 없어 검증셋
숫자가 없었고(README 가 빈 템플릿이었다), 이 스크립트가 처음으로 재현 가능한 기준선을 만든다.

    python3 -m src.sms_eval                    # adapters/sms 를 task=sms 로
    python3 -m src.sms_eval --task sms2        # 후보 어댑터를 adapters/sms2 에 두고 비교할 때

**이 점수를 성능 근거로 쓰지 말 것.** 문장을 지은 쪽이 학습 틀도 썼다(같은 사람). 용도는
어댑터를 바꿨을 때 이전과 나란히 비교하는 것이다. 정답 파일은 학습 저장소에 있다 —
`../Voice-Detection/eval/sms_cases.jsonl`, `sms_scenarios.jsonl`.

시나리오는 조각별로 따로 넣는다. 앱이 그렇게 하기 때문이다(채널마다 판정, 세션은 겹침만 센다).
마지막 줄의 "결합"은 조각을 `[문자] … [통화] …` 로 이어 한 번에 넣은 것으로, 세션 단위
재판정을 하면 어떻게 될지 미리 보는 자리다.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "Voice-Detection" / "eval"
CASES = SIBLING / "sms_cases.jsonl"
SCENARIOS = SIBLING / "sms_scenarios.jsonl"


def analyze(url, text, task, timeout):
    req = urllib.request.Request(
        f"{url}/analyze",
        data=json.dumps({"text": text, "task": task}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def tier(risk, stage):
    """앱 `BackendClassificationClient.signalOf` 와 같은 문턱. 여기서 다시 정의하지 않고 베낀다."""
    if risk >= 80:
        return "경보"
    if risk >= 50:
        return "경보" if stage else "경보우려"
    if risk >= 20:
        return "주의보" if stage else "예보"
    return "정상"


def run_cases(url, task, timeout):
    rows = [json.loads(line) for line in CASES.open(encoding="utf-8")]
    print(f"{'id':>12} {'정답':>4} {'위험도':>7} {'단계':>4} {'등급':>6}  본문")
    print("-" * 80)
    out = []
    for r in rows:
        res = analyze(url, r["text"], task, timeout)
        t = tier(res["risk"], res.get("stage"))
        wrong = (r["label"] == 1 and res["risk"] < 20) or (r["label"] == 0 and res["risk"] >= 20)
        out.append((r, res["risk"], res.get("stage"), t))
        print(f"{r['id']:>12} {'사기' if r['label'] else '정상':>4} {res['risk']:7.1f} "
              f"{str(res.get('stage') or '-'):>4} {t:>6}  {r['text'][:34]}{'  ←' if wrong else ''}")

    print("\n== 낱개 요약 ==")
    for label, name in ((0, "정상"), (1, "사기")):
        part = [o for o in out if o[0]["label"] == label]
        risks = sorted(o[1] for o in part)
        tiers = {}
        for o in part:
            tiers[o[3]] = tiers.get(o[3], 0) + 1
        print(f"{name} n={len(part)}  중앙값 {risks[len(risks)//2]:.1f}  "
              f"min {risks[0]:.1f}  max {risks[-1]:.1f}  등급 {tiers}")
    for th in (20, 50, 80):
        tp = sum(1 for o in out if o[0]["label"] == 1 and o[1] >= th)
        fp = sum(1 for o in out if o[0]["label"] == 0 and o[1] >= th)
        n1 = sum(1 for o in out if o[0]["label"] == 1)
        n0 = len(out) - n1
        print(f"문턱 {th:2}: 사기 탐지 {tp}/{n1}   정상 오탐 {fp}/{n0}")
    # 20~80 사이(애매한 구간)에 몇 건이 있는지 — 보정이 됐다면 여기 무언가 있어야 한다
    mid = sum(1 for o in out if 20 <= o[1] < 80)
    print(f"20~80 구간 {mid}건 / {len(out)}건  (0 이면 확률이 양극화된 것이다)")
    return out


def run_scenarios(url, task, timeout):
    rows = [json.loads(line) for line in SCENARIOS.open(encoding="utf-8")]
    print("\n== 다채널 시나리오 (조각별 → 결합) ==")
    caught = 0
    for s in rows:
        print(f"\n# {s['name']}  ({'사기' if s['label'] else '정상'})")
        pieces = []
        for i, st in enumerate(s["steps"], 1):
            # 통화 조각은 통화 어댑터로, 문자 조각은 이 태스크로 — 앱과 같다.
            t = "voice" if st["channel"] == "voice" else task
            res = analyze(url, st["text"], t, timeout)
            pieces.append(res["risk"])
            print(f"  {i}. [{st['channel']:5}] {res['risk']:6.1f} stage={res.get('stage')}"
                  f" → {tier(res['risk'], res.get('stage')):4} | {st['text'][:36].replace(chr(10), ' ')}…")
        joined = "\n".join(f"[{'통화' if st['channel']=='voice' else '문자'}] {st['text']}" for st in s["steps"])
        res = analyze(url, joined, task, timeout)
        print(f"  ↳ 결합 [{task}] {res['risk']:6.1f} stage={res.get('stage')} → {tier(res['risk'], res.get('stage'))}")
        if s["label"] == 1 and all(p >= 20 for p in pieces):
            caught += 1
        if s["label"] == 1 and any(p < 20 for p in pieces):
            print("     조각 하나가 정상 판정 — 앱에서는 격상되지 않는다")
    n1 = sum(1 for s in rows if s["label"] == 1)
    print(f"\n사기 시나리오 {n1}건 중 모든 조각이 예보 이상: {caught}건 (= 앱이 격상까지 하는 건수)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--task", default="sms", help="adapters/<이름> 폴더 이름")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-scenarios", action="store_true")
    args = parser.parse_args()

    try:
        run_cases(args.url, args.task, args.timeout)
        if not args.no_scenarios:
            run_scenarios(args.url, args.task, args.timeout)
    except urllib.error.URLError as e:
        raise SystemExit(f"서버에 닿지 못했습니다 ({args.url}) — 먼저 띄우세요.\n  {e}")

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Detection-Server** — 학습된 어댑터를 받아 위험도와 근거를 돌려주는 추론 서버.

안드로이드 앱(`../demo`)이 통화 전사본을 보내면 0~100 위험도를 준다. 학습 저장소
(`../Voice-Detection`)와 분리한 이유는 주기가 다르기 때문이다 — 학습은 가끔 돌리는 오프라인
배치이고 서버는 항상 떠 있어야 한다.

```
전사본  ──▶  어댑터 ON   ──▶  P("예") = 0.93  ──▶  위험도 93
        └─▶  어댑터 OFF  ──▶  원본 Gemma가 원문을 인용해 근거 작성
```

**이 저장소는 데이터를 들고 다니지 않는다.** 어댑터 폴더만 있으면 된다. 학습 코드도 참조하지
않는다 — 프롬프트는 어댑터에 딸려 온 `prompt.json`에서 재구성한다.

> 아직 git 저장소가 아니다. `git init` 후 원격을 붙일 것.

## Critical Rules

- **학습 저장소의 코드를 import하거나 복사하지 말 것.** 프롬프트는 `prompt.json`에서
  재구성한다. 코드를 복사하면 언젠가 갈라지고, 갈라지는 순간 **에러 없이 확률만 조용히
  틀어진다.** 이 저장소가 존재하는 방식 자체가 그 사고를 막는 장치다.
- **베이스 모델은 프로세스당 한 번만 올린다.** 어댑터는 여러 개 붙일 수 있다
  (`load_adapter` / `set_adapter`). 태스크마다 서버를 띄우면 8GB짜리 베이스를 중복 적재해
  24GB 맥에서 터진다.
- **모델에게 숫자를 묻지 말 것.** 위험도는 정답 토큰 자리의 로짓에서 읽는다. "몇 %냐"고
  물어 뱉게 하면 학습으로 점검된 적 없는 값이 나온다.
  **진행 단계(1~4)도 같은 함정이었다** — `assess()`가 생성으로 단계를 받는데, 실측에서 거의
  전부 3으로 쏠렸다(`../Voice-Detection/docs/RESULTS.md`). 근거 인용은 맞게 하므로 독해가
  아니라 분류가 안 되는 것이다. **`stage` 값은 아직 신뢰하지 말 것.** 규칙 기반으로 바꾸는
  것이 다음 수다.
- **근거 프롬프트는 판정에 따라 갈라야 한다.** `"이 통화는 보이스피싱으로 판정되었다"`를
  무조건 전제로 깔면 정상 통화에도 억지 근거를 만든다 — 배송 지연 사과를 "감정에 호소하는
  사기 수법"으로 둔갑시키는 것을 실제로 확인했다. 모델은 주어진 전제를 의심하지 않는다.
- **순전파는 락으로 직렬화한다.** GPU 하나에 동시 진입시키지 말 것.
- **어댑터 가중치를 커밋하지 말 것.** 학습 저장소의 산출물이다. `prompt.json`과
  `calibration.json`만 예외로 남긴다 — 작고, 서버 동작을 규정하는 값이다.
- **`BASE_MODEL` 환경변수는 같은 가중치의 미러에만 쓸 것.** 다른 모델을 넣으면 어댑터가
  엉뚱한 곳에 붙어 확률이 망가진다. 바꿨다면 반드시 알려진 값으로 대조할 것(아래 검증 참고).
- **주석은 "왜"만 쓴다.**

## Architecture

```markdown
Detection-Server/
├── app.py                  # FastAPI. lifespan에서 모델 1회 적재
│                           #   POST /analyze   {text, task, reason} → {risk, reason}
│                           #   GET  /health    적재된 어댑터와 온도
│
├── engine.py               # 추론 본체
│   ├── class Task          #   어댑터 하나 + 그 계약(prompt.json, calibration.json)
│   │   └── prompt_ids()    #   학습 때와 동일한 프롬프트 재구성  ← 가장 중요한 함수
│   └── class Engine
│       ├── _placement()    #   CUDA → 4bit / MPS → bf16 / 그 외 → fp32
│       ├── risk()          #   어댑터 ON. 로짓에서 P("예"), 온도로 나눔. 생성 안 함
│       ├── assess()        #   disable_adapter() 블록에서 원본 Gemma가 단계·근거 작성
│       └── _parse()        #   `단계: 3 / 근거: …` 를 뜯는다. 형식 위반을 전제로 짠다
│
├── adapters/               # (gitignore — prompt.json·calibration.json만 예외)
│   └── voice/              #   폴더 이름이 곧 API의 task 값이 된다
│       ├── adapter_model.safetensors
│       ├── prompt.json         지시문·입력 상한·정답 후보  ← 계약
│       ├── calibration.json    온도 1.370
│       └── tokenizer.json / chat_template.jinja
│
├── docs/
│   └── api.md              # 엔드포인트·계약 형식·어댑터 추가 절차
│
└── requirements.txt        # bitsandbytes 없음 (맥에서 못 씀)
```

## Tech Stack

- Python 3.13 / FastAPI + uvicorn
- `torch` (MPS) / `transformers>=4.50` / `peft` / `accelerate`
- **`bitsandbytes`는 requirements에 없다.** CUDA 전용이라 맥에서 설치해도 못 쓴다.
  GPU 서버에 올릴 때만 따로 추가한다.
- 베이스 모델은 계약(`prompt.json`)이 지정한다. 현재 `google/gemma-3-4b-it`
- 메모리 — 베이스 bf16 8.0GB + 어댑터 0.125GB/개

## Build & Test Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p adapters
cp -R ../Voice-Detection/adapter adapters/voice     # checkpoint-* 는 빼고 복사할 것

# 베이스 모델이 승인 필요 저장소라 토큰이 없으면 401로 죽는다.
# huggingface-cli login 을 했다면 환경변수 없이 그냥 uvicorn 만 실행하면 된다.
BASE_MODEL=unsloth/gemma-3-4b-it uvicorn app:app --host 0.0.0.0 --port 8000
```

`0.0.0.0`이어야 다른 기기에서 닿는다. 모델 적재에 수십 초가 걸리므로 `준비 완료` 로그를 보고
요청할 것.

```bash
curl -s localhost:8000/health | python3 -m json.tool
curl -s localhost:8000/analyze -H 'Content-Type: application/json' \
     -d '{"text":"…전사본…","task":"voice"}' | python3 -m json.tool
```

**에뮬레이터·실기기에서 붙일 때는 `adb reverse`를 쓴다.**

```bash
adb reverse tcp:8000 tcp:8000     # 기기의 localhost:8000 → 맥의 8000
```

`10.0.2.2` 직결은 맥 방화벽이 TCP를 막아 타임아웃난다(ICMP는 통과해서 ping은 성공한다).

**자동화된 테스트가 없다.** 어댑터를 바꾸거나 `BASE_MODEL`을 손댔다면 **알려진 값으로 대조**할
것. `../Voice-Detection/docs/RESULTS.md`에 Colab 실측값이 있다.

```
vishing_120  →  99.9      낮은 건과 높은 건을 함께 볼 것.
vishing_384  →   2.5      높은 건만 맞추면 "무조건 높게 뱉는 상태"도 통과한다.
```

## Domain Context

- **어댑터(adapter)** — LoRA 가중치 125MB. 원본 Gemma는 얼려두고 이것만 갈아끼운다.
  폴더 이름이 API의 `task` 값이 된다(`adapters/voice` → `"task": "voice"`).
- **계약(prompt.json)** — 지시문·입력 토큰 상한·정답 후보(`예`/`아니오`)·베이스 모델 이름.
  학습 스크립트가 어댑터와 함께 저장한다. 서버는 이것만 보고 프롬프트를 만든다.
- **온도(calibration.json)** — 과확신을 누르는 스칼라. 검증셋에서 NLL을 최소화해 구한 값이라
  "87점 = 실제로 87% 확률"이 성립한다. 없으면 보정 없이 간다.
- **위험도 vs 단계** — 위험도는 "피싱이 맞나"이고 로짓에서 읽는다. 단계는 "지금 어디까지
  왔나"이고 원본 Gemma가 생성한다. 같은 99%라도 사칭만 한 상태와 계좌를 부르는 상태는
  사용자가 할 행동이 다르다. 다만 현재 단계 판정은 부정확하다(위 Critical Rules 참고).
- **비용이 다르다** — 위험도는 순전파 한 번(0.3초), 단계·근거는 160토큰 생성(MPS에서 15초
  이상). 그래서 API에서 분리했고, 앱은 위험도가 문턱을 넘을 때만 두 번째 요청을 보낸다.
- **어댑터 OFF** — `disable_adapter()` 블록 안에서는 원본 Gemma다. 근거 작성은 학습시키지 않은
  능력이고, 원문을 읽고 인용하는 건 대화형 모델이 이미 잘한다.

## Coding Conventions

- 모든 주석·docstring·로그는 **한국어**. 커밋 메시지는 영어.
- docstring은 **결정의 근거**를 적는다 (예: 왜 근거 프롬프트가 두 벌인지).
- 무거운 import(`torch`, `transformers`, `peft`)는 함수 안에서 한다.
- 설정은 환경변수로 받는다 — `BASE_MODEL`, `DEVICE`. 코드에 하드코딩하지 않는다.
- 실패는 `HTTPException`으로 상태 코드와 함께 올린다. 조용히 기본값으로 떨어지지 않는다.
- 계약이 어긋나면(정답 후보 첫 토큰이 같은 등) **적재 시점에 죽인다.** 잘못된 확률을 내는
  것보다 안 뜨는 편이 낫다.

## Key Patterns

- **파일로 고정한 계약** — 저장소가 갈라져도 프롬프트는 갈라지지 않는다. 코드 공유 대신
  데이터 공유를 택했다.
- **베이스 1개 + 어댑터 N개** — `PeftModel.from_pretrained(..., adapter_name=)`로 첫 개를 붙이고
  나머지는 `load_adapter`. 요청마다 `set_adapter(task)`로 전환한다.
- **첫 토큰만 비교** — `아니오`는 3토큰이지만 첫 토큰(`아`)만 `예`와 다르면 된다. 한 자리에서
  softmax를 취해야 그 값이 곧 확률이다.
- **배치 위치 자동 선택** — `_placement()`가 CUDA/MPS/CPU를 보고 양자화 여부와 dtype을 정한다.
  맥에서 4bit를 시도하면 bitsandbytes가 없어 죽는다.
- **근거는 실패해도 판정을 살린다** — 호출부(앱)가 근거 생성 실패를 잡아 위험도만 쓴다.
  느린 부가 기능이 핵심 판정을 막지 않게 한다.

## Reference Docs

- `docs/api.md` — 엔드포인트 스펙, 계약 파일 형식, 어댑터 추가 절차, 오류 응답.
- `README.md` — 설치·실행 요약과 현재 한계.
- `../Voice-Detection/docs/METHOD.md` — 왜 확률을 로짓에서 읽는가. 이 서버가 하는 일의 근거.
- `../Voice-Detection/docs/RESULTS.md` — 알려진 실측값. 이식·회귀 검증의 기준선.
- `../Voice-Detection/CLAUDE.md` — 어댑터를 만드는 쪽.

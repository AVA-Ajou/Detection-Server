# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Detection-Server** — 학습된 어댑터를 받아 위험도와 근거를 돌려주는 추론 서버.

안드로이드 앱(`../demo`)이 통화 전사본을 보내면 0~100 위험도를 준다. 학습 저장소
(`../Voice-Detection`)와 분리한 이유는 주기가 다르기 때문이다 — 학습은 가끔 돌리는 오프라인
배치이고 서버는 항상 떠 있어야 한다.

```
통화 음성  ──▶  어댑터 OFF  ──▶  전사본                    (POST /transcribe)
전사본    ──▶  어댑터 ON   ──▶  P("예") = 0.93 → 위험도 93  (POST /analyze)
          └─▶  어댑터 OFF  ──▶  원본 Gemma가 원문을 인용해 근거 작성
```

베이스가 **Gemma 4 E2B**라 음성을 직접 받는다. 그래서 전사(STT)도 이 서버가 한다 — 앱이
Groq Whisper 같은 외부 API에 통화 음성을 넘기지 않아도 된다. 모델은 여전히 하나다.

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
  **진행 단계도 같은 함정이었다.** 예전에는 `assess()`가 생성으로 단계를 받았는데 한쪽으로
  쏠렸다. 지금은 `signals.py` 의 규칙이 준다 — 정답지 36건에서 규칙 91.7% / 생성 33.3%
  (`stage_eval.py`). **단계를 다시 프롬프트로 옮기지 말 것.**
- **단계를 손대면 `stage_eval.py` 를 돌릴 것.** 정규식은 고치기 쉬워서 눈대중으로 바꾸기
  쉬운데, 그러면 "나아 보인다"에서 멈춘다. 정답지는 규칙보다 **먼저** 만들어졌다
  (`../Voice-Detection/eval/stage_gold.jsonl`) — 규칙에 맞춰 정답을 고치지 말 것.
- **규칙이 신호를 못 찾으면 `stage` 는 `null` 이다. 1단계로 올려 찍지 말 것.** 증거 없이
  단계를 붙이는 것이 생성 모델이 하던 실수다. 피싱의 약 15%가 여기 해당한다.
- **근거 프롬프트는 판정에 따라 갈라야 한다.** `"이 통화는 보이스피싱으로 판정되었다"`를
  무조건 전제로 깔면 정상 통화에도 억지 근거를 만든다 — 배송 지연 사과를 "감정에 호소하는
  사기 수법"으로 둔갑시키는 것을 실제로 확인했다. 모델은 주어진 전제를 의심하지 않는다.
- **순전파는 락으로 직렬화한다.** GPU 하나에 동시 진입시키지 말 것.
- **어댑터 가중치를 커밋하지 말 것.** 학습 저장소의 산출물이다. `prompt.json`과
  `calibration.json`만 예외로 남긴다 — 작고, 서버 동작을 규정하는 값이다.
- **`BASE_MODEL` 환경변수는 같은 가중치의 미러에만 쓸 것.** 다른 모델을 넣으면 어댑터가
  엉뚱한 곳에 붙어 확률이 망가진다. 바꿨다면 반드시 알려진 값으로 대조할 것(아래 검증 참고).
- **`adapters/` 아래 어댑터는 베이스가 전부 같아야 한다.** 서버는 베이스를 하나만 올리므로
  베이스가 다른 어댑터를 섞으면 한쪽이 엉뚱한 모델에 붙는다 — **에러 없이 확률만 틀어진다.**
  구형 Gemma 3 어댑터는 학습 저장소(`../Voice-Detection/adapter/`)에 두고 여기 넣지 말 것.
- **주석은 "왜"만 쓴다.**

## Architecture

```markdown
Detection-Server/
├── app.py                  # FastAPI 진입점. **루트에 남긴다** — `uvicorn app:app` 이 그대로 돈다
│                           #   POST /analyze     {text, task, reason} → {risk, reason}
│                           #   POST /transcribe  multipart 오디오 → {text}
│                           #   GET  /health      적재된 어댑터·온도·오디오 가능 여부
│   └── _to_wav()           #   afconvert 로 m4a → 16kHz 모노 wav (macOS 전용)
│
│                           # src/ 는 패키지다. `app.py` 가 `from src.engine import Engine`
│                           # 으로 가져가고, 내부에서는 상대 import(`from . import signals`)를
│                           # 쓴다. 그래서 스크립트는 `python3 -m src.<이름>` 으로 돌린다.
│
├── src/engine.py           # 추론 본체
│   ├── class Task          #   어댑터 하나 + 그 계약(prompt.json, calibration.json)
│   │   └── prompt_ids()    #   학습 때와 동일한 프롬프트 재구성  ← 가장 중요한 함수
│   └── class Engine
│       ├── _placement()    #   CUDA → 4bit / MPS → bf16 / 그 외 → fp32
│       ├── _load_processor() # 오디오 프로세서. audio_config 없으면 None → /transcribe 501
│       ├── risk()          #   어댑터 ON. 로짓에서 P("예"), 온도로 나눔. 생성 안 함
│       ├── transcribe()    #   disable_adapter() 블록에서 원본 Gemma가 음성을 받아적음
│       ├── stage()         #   signals.stage_of() 를 감싼다. 정규식이라 비용 0
│       ├── explain()       #   근거 문장 생성. 앱은 안 쓴다 — reason:true 일 때만 돈다
│       └── _clean()        #   생성물에서 근거 문장만 남긴다. 형식 위반을 전제로 짠다
│
├── src/signals.py          # 진행 단계를 정규식으로 읽는다. 모델을 쓰지 않는다
│   ├── SIGNALS             #   (단계, 이름, 패턴). 2·3단계는 요구/질문 문형까지 요구한다
│   └── stage_of()          #   발동한 신호 중 최대 단계 + 인용구. 없으면 (None, [])
│
├── src/stage_eval.py       # 정답지에 규칙과 생성 모델을 나란히 걸어 정확도를 낸다
│                           #   python3 -m src.stage_eval --llm
│
├── adapters/               # (gitignore — prompt.json·calibration.json만 예외)
│   └── voice/              #   폴더 이름이 곧 API의 task 값이 된다
│       ├── adapter_model.safetensors
│       ├── prompt.json         지시문·입력 상한·정답 후보  ← 계약
│       ├── calibration.json    온도 3.136
│       └── tokenizer.json / chat_template.jinja
│
├── docs/
│   └── api.md              # 엔드포인트·계약 형식·어댑터 추가 절차
│
└── requirements.txt        # bitsandbytes 없음 (맥에서 못 씀)
                            # librosa·soundfile·pillow·torchvision 은 프로세서가 요구한다
```

## Tech Stack

- Python 3.13 / FastAPI + uvicorn
- `torch` (MPS) / `transformers>=4.50` / `peft` / `accelerate`
- `librosa` / `soundfile` / `pillow` / `torchvision` — Gemma 4의 **프로세서**가 요구한다.
  텍스트만 쓸 생각이어도 프로세서를 못 열면 `/transcribe`가 501로 죽는다.
- **`bitsandbytes`는 requirements에 없다.** CUDA 전용이라 맥에서 설치해도 못 쓴다.
  GPU 서버에 올릴 때만 따로 추가한다.
- 베이스 모델은 계약(`prompt.json`)이 지정한다. 현재 `google/gemma-4-E2B-it`
  (게이트되지 않아 토큰 없이 받을 수 있다)
- 메모리 — 베이스 bf16 약 10GB + 어댑터 0.09GB/개
- **`afconvert`(macOS 내장)에 의존한다.** 앱이 보내는 `.m4a`를 librosa가 못 읽어 wav로
  바꾼다. 리눅스에 올릴 때 `app._to_wav()`를 ffmpeg 등으로 갈아끼워야 하는 유일한 지점이다.

## Build & Test Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p adapters
# checkpoint-* 는 빼고 복사할 것. 어댑터는 **하나만** 넣는다 (베이스가 같아야 한다)
rsync -a --exclude 'checkpoint-*' ../Voice-Detection/adapter-gemma4/ adapters/voice/
cat adapters/voice/prompt.json      # base_model 이 gemma-4-E2B-it 인지 확인

# 맥에는 unsloth 미러가 받아져 있어 그쪽을 가리킨다. 정품을 쓰려면 환경변수 없이 실행하면
# 되는데(게이트 아님) 9.6GB를 새로 받는다.
BASE_MODEL=unsloth/gemma-4-E2B-it uvicorn app:app --host 0.0.0.0 --port 8000
```

`0.0.0.0`이어야 다른 기기에서 닿는다. 모델 적재에 수십 초가 걸리므로 `준비 완료` 로그를 보고
요청할 것. 그 위에 **`오디오 입력  가능`**이 떠야 `/transcribe`가 동작한다 — `불가`면 위
오디오 의존성이 빠진 것이다.

```bash
curl -s localhost:8000/health | python3 -m json.tool
curl -s localhost:8000/analyze -H 'Content-Type: application/json' \
     -d '{"text":"…전사본…","task":"voice"}' | python3 -m json.tool
curl -s -F file=@통화.m4a localhost:8000/transcribe | python3 -m json.tool
```

**에뮬레이터·실기기에서 붙일 때는 `adb reverse`를 쓴다.**

```bash
adb reverse tcp:8000 tcp:8000     # 기기의 localhost:8000 → 맥의 8000
```

`10.0.2.2` 직결은 맥 방화벽이 TCP를 막아 타임아웃난다(ICMP는 통과해서 ping은 성공한다).

**단계 규칙에는 정답지가 있다.** `signals.py`를 고쳤다면 반드시 돌릴 것 — 모델을 안 올려서
몇 초면 끝난다.

```bash
python3 -m src.stage_eval                  # 규칙만. 현재 91.7%
python3 -m src.stage_eval --show-misses    # 틀린 건과 규칙이 본 근거
python3 -m src.stage_eval --llm            # 생성 모델 기준선까지 (모델 적재, 수 분)
```

`--llm`을 돌릴 때는 **서버를 먼저 내릴 것.** 모델을 하나 더 올려 24GB 맥에서 메모리가 터진다.

**그 밖에는 자동화된 테스트가 없다.** 어댑터를 바꾸거나 `BASE_MODEL`을 손댔다면 **알려진
값으로 대조**할 것. `../Voice-Detection/docs/RESULTS.md`에 Colab 실측값이 있다.

```
vishing_120  →  99.9      낮은 건과 높은 건을 함께 볼 것.
vishing_384  →   7.5      높은 건만 맞추면 "무조건 높게 뱉는 상태"도 통과한다.
```

맥(bf16)과 Colab(4bit)은 수치 방식이 달라 완전 일치하지 않는다. **판정이 뒤집히지 않으면
통과**로 본다.

## Domain Context

- **어댑터(adapter)** — LoRA 가중치 92MB. 원본 Gemma는 얼려두고 이것만 갈아끼운다.
  폴더 이름이 API의 `task` 값이 된다(`adapters/voice` → `"task": "voice"`).
- **계약(prompt.json)** — 지시문·입력 토큰 상한·정답 후보(`예`/`아니오`)·베이스 모델 이름.
  학습 스크립트가 어댑터와 함께 저장한다. 서버는 이것만 보고 프롬프트를 만든다.
- **온도(calibration.json)** — 과확신을 누르는 스칼라. 검증셋에서 NLL을 최소화해 구한 값이라
  "87점 = 실제로 87% 확률"이 성립한다. 없으면 보정 없이 간다.
- **위험도 vs 단계** — 위험도는 "피싱이 맞나"이고 **로짓**에서 읽는다. 단계는 "지금 어디까지
  왔나"이고 **정규식**이 읽는다. 같은 99%라도 압박만 한 상태와 계좌로 입금하라는 상태는
  사용자가 할 행동이 다르다. **둘 다 생성하지 않는다** — 세 값 중 모델이 생성하는 것은
  근거 문장 하나뿐이다.
- **화면에 나가는 것은 단계다** — 앱은 위험도 숫자를 띄우지 않는다. 값이 0 아니면 100으로
  갈려 정보가 없기 때문이다. 원본은 DB와 로그에 남는다.
- **비용이 다르다** — 위험도는 순전파 한 번(0.4~2초), 단계는 정규식이라 0초, 근거 문장은
  160토큰 생성(MPS에서 15초 이상), 전사는 25초짜리 통화에 약 10초.
  **위험도와 단계는 한 응답에 함께 나간다** — 예전에는 단계가 근거 생성에 딸려 있어서
  앱이 요청을 두 번 보내고 판정 한 건에 16초를 썼다. 지금은 한 번, 2초 이하다.
- **어댑터 OFF** — `disable_adapter()` 블록 안에서는 원본 Gemma다. 근거 작성도 전사도
  학습시키지 않은 능력이고, 원문을 읽고 인용하거나 음성을 받아적는 건 대화형 멀티모달 모델이
  이미 잘한다. **학습으로 얻은 것은 확률 하나뿐이고, 나머지는 원래 있던 것을 쓴다.**

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

# Detection-Server

학습된 어댑터를 받아 위험도를 돌려주는 추론 서버. 앱(`../demo`)이 호출한다.

학습 저장소(`../Voice-Detection`)와 분리한 이유는 주기가 다르기 때문이다 — 학습은 가끔
돌리는 오프라인 배치이고, 서버는 항상 떠 있어야 한다. 이 저장소는 **데이터를 들고 다니지
않는다.** 어댑터 폴더만 있으면 된다.

```
전사본  ──▶  어댑터 ON   ──▶  P("예") = 0.93  ──▶  위험도 93
        └─▶  어댑터 OFF  ──▶  원본 Gemma가 원문을 인용해 근거 작성
```

## 왜 서버가 하나인가

LoRA는 원본 가중치를 건드리지 않으므로, 베이스 하나에 어댑터를 여러 개 얹을 수 있다.

```
   Gemma 3 4B    8.00GB   ← 한 번만 올린다
   voice 어댑터   0.125GB
   sms 어댑터     0.125GB  (나중에)
   ───────────────────────
                 8.25GB
```

태스크마다 서버를 따로 띄우면 8GB짜리 베이스를 중복 적재해 24GB 맥에서 위험해진다.

## 프롬프트 계약 — 이 저장소의 전제

학습 때와 추론 때 프롬프트가 **한 글자라도 다르면 에러 없이 확률만 조용히 틀어진다.**
그래서 이 저장소는 학습 코드를 복사해 오지 않고, 어댑터에 딸려 온 `prompt.json`을 읽어
프롬프트를 재구성한다.

```
adapters/voice/
├── adapter_model.safetensors    가중치 (커밋 안 함)
├── adapter_config.json
├── prompt.json                  지시문 / 입력 상한 / 정답 후보   ← 계약
├── calibration.json             온도. 없으면 보정 없이 간다
└── chat_template.jinja 등        토크나이저
```

학습 쪽에서 `train_lora.py`가 전부 함께 저장한다. 이미 학습이 끝난 어댑터라면
`python3 common.py --write-contract adapter/` 로 계약만 붙일 수 있다.

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

어댑터를 넣는다. 폴더 이름이 곧 `task` 이름이 된다.

```bash
mkdir -p adapters
cp -R ../Voice-Detection/adapter adapters/voice
```

## 실행

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

베이스 모델 `google/gemma-3-4b-it`은 **승인이 필요한 저장소**다. 허깅페이스 토큰이 없으면
401로 죽는다. 로그인하거나, 같은 가중치의 공개 미러를 지정한다.

```bash
huggingface-cli login                        # ① 정품
BASE_MODEL=unsloth/gemma-3-4b-it \
    uvicorn app:app --host 0.0.0.0 --port 8000   # ② 미러 (승인 불필요)
```

미러를 썼다면 **알려진 값으로 대조할 것** — `../Voice-Detection/docs/RESULTS.md`에 기준선이 있다.

`0.0.0.0` 이어야 같은 와이파이의 폰에서 닿는다. 기본값이면 맥 안에서만 보인다.
모델을 올리는 데 수십 초가 걸리므로 `준비 완료` 로그를 보고 요청할 것.

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

```json
{"device": "mps",
 "tasks": {"voice": {"temperature": 1.37, "calibrated": true, "max_text_tokens": 1024}}}
```

## API

```
POST /analyze   {"text": "...", "task": "voice", "reason": false}
             →  {"task": "voice", "risk": 93.2, "probability": 0.932, "elapsed_ms": 420}
```

`reason`을 켜면 근거 문장이 붙지만 256토큰을 생성해야 해서 수십 배 느리다. 앱은 위험도를
먼저 받고, 사용자가 "왜?"를 눌렀을 때 따로 요청하는 흐름을 쓴다.

## 환경 변수

| | |
|---|---|
| `DEVICE=cpu` | GPU를 무시하고 CPU로 강제. 디버깅용 |

## 아직 안 되는 것

- **위험도가 사실상 0 아니면 100이다.** 학습·검증이 통화 **전체**를 봤기 때문으로 보인다.
  실전은 진행 중인 통화의 앞부분만 들어오므로 다를 수 있고, `Voice-Detection/evaluate.py
  --prefix` 로 확인 중이다. 그때까지 앱은 중간 등급을 기대하면 안 된다.
- **다채널 융합은 앱 몫이다.** 통화 위험도를 로그오즈로 바꿔 들고 있다가 문자 신호가 나오면
  더하는 방식 — `Voice-Detection/docs/METHOD.md` 6절.

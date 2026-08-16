# Detection-Server

학습된 어댑터를 받아 위험도를 돌려주는 추론 서버. 앱(`../AVA-app`)이 호출한다.

학습 저장소(`../Voice-Detection`)와 분리한 이유는 주기가 다르기 때문이다 — 학습은 가끔
돌리는 오프라인 배치이고, 서버는 항상 떠 있어야 한다. 이 저장소는 **데이터를 들고 다니지
않는다.** 어댑터 폴더만 있으면 된다.

```
통화 음성  ──▶  어댑터 OFF  ──▶  전사본                    (POST /transcribe)
전사본    ──▶  어댑터 ON   ──▶  P("예") = 0.93 → 위험도 93  (POST /analyze)
          └─▶  어댑터 OFF  ──▶  원본 Gemma가 원문을 인용해 근거 작성
```

## 왜 모델이 하나인가

베이스가 **Gemma 4 E2B**라 음성을 직접 받는다. 판정·근거·전사를 같은 가중치가 처리하므로
서버에 모델을 하나만 올린다. 앱이 통화 음성을 Groq Whisper 같은 외부 API로 내보내던 경로가
이걸로 사라졌다.

LoRA는 원본 가중치를 건드리지 않으므로, 베이스 하나에 어댑터를 여러 개 얹을 수도 있다.

```
   Gemma 4 E2B   10.0GB   ← 한 번만 올린다
   voice 어댑터   0.09GB
   sms 어댑터     0.09GB  (나중에)
   ───────────────────────
                 10.2GB
```

태스크마다 서버를 따로 띄우면 10GB짜리 베이스를 중복 적재해 24GB 맥에서 위험해진다.

**단, 어댑터들의 베이스가 같아야 한다.** 서버는 베이스를 하나만 올리므로 Gemma 3 어댑터를
섞어 넣으면 한쪽이 엉뚱한 모델에 붙어 **에러 없이 확률만 틀어진다.**

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
`python3 src/common.py --write-contract adapter/` 로 계약만 붙일 수 있다.

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

어댑터를 넣는다. 폴더 이름이 곧 `task` 이름이 된다. **하나만 넣는다** — 베이스가 다른
어댑터를 섞으면 안 된다.

```bash
mkdir -p adapters
rsync -a --exclude 'checkpoint-*' ../Voice-Detection/adapter-gemma4/ adapters/voice/
cat adapters/voice/prompt.json      # base_model 이 gemma-4-E2B-it 인지 확인
```

## 실행

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

`google/gemma-4-E2B-it`은 게이트되지 않아 토큰 없이 받는다(9.6GB). 이미 받아둔 미러가 있으면
그쪽을 가리켜 재다운로드를 피한다.

```bash
BASE_MODEL=unsloth/gemma-4-E2B-it uvicorn app:app --host 0.0.0.0 --port 8000
```

미러를 썼다면 **알려진 값으로 대조할 것** — `../Voice-Detection/docs/RESULTS.md`에 기준선이 있다.

`0.0.0.0` 이어야 같은 와이파이의 폰에서 닿는다. 기본값이면 맥 안에서만 보인다.
모델을 올리는 데 수십 초가 걸리므로 `준비 완료` 로그를 보고 요청할 것. 그 위에 이 줄이
떠야 `/transcribe`가 동작한다.

```
오디오 입력  가능
```

`불가`면 프로세서를 못 연 것이다 — `librosa`·`soundfile`·`pillow`·`torchvision`이 있는지 본다.

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

```json
{"device": "mps",
 "audio": true,
 "tasks": {"voice": {"temperature": 3.136, "calibrated": true, "max_text_tokens": 1024,
                     "base_model": "google/gemma-4-E2B-it"}}}
```

## API

```
POST /analyze     {"text": "...", "task": "voice", "reason": false}
               →  {"task": "voice", "risk": 93.2, "probability": 0.932, "elapsed_ms": 420}

POST /transcribe  multipart: file=@통화.m4a, task=voice
               →  {"text": "여보세요 …", "elapsed_ms": 9940}
```

`reason`을 켜면 근거 문장과 **진행 단계**가 붙는다. 근거는 256토큰을 생성해야 해서 수십 배
느리고, 단계는 정규식이라 공짜다. 앱은 `/analyze` 한 번으로 위험도와 단계를 함께 받아
**둘을 같이 보고** 등급을 정한다 — 66 이상은 경보, 33~66은 단계가 잡힐 때만 주의보다.

```
1 압박·유인   2 정보·계좌   3 이체 지시
```

**단계는 모델이 아니라 `signals.py` 가 뽑는다.** 정답지 36건에서 규칙 91.7% / 생성 33.3%
(`python3 -m src.stage_eval --llm` 으로 재현). 생성 모델은 1단계를 한 번도 예측하지 못했다.

전사는 어댑터를 끈 원본 Gemma 4가 한다. `.m4a`는 macOS 내장 `afconvert`로 16kHz 모노 wav로
바꿔 넘긴다 — **리눅스에 올릴 때 `app._to_wav()`를 ffmpeg 등으로 갈아끼워야 한다.**
자세한 스펙은 `docs/api.md`.

## 환경 변수

| | |
|---|---|
| `BASE_MODEL` | 계약의 `base_model`을 덮어쓴다. **같은 가중치의 미러에만 쓸 것** |
| `DEVICE=cpu` | GPU를 무시하고 CPU로 강제. 디버깅용 |

## 아직 안 되는 것

- **정당한 금융 아웃바운드 통화에 약하다.** 학습셋에 "돈 얘기 + 기관이 먼저 건" 정상 통화가
  711건 중 6건뿐이라(피싱은 204건), 은행이 먼저 걸어온 정상 통화를 피싱으로 본다.
  어려운 정상 20건에서 **오탐률 10.0%** — 검증셋 213건에서는 0.0%였다.
  `Voice-Detection/eval_hard_normal.py` 로 잰다. 학습셋 보강이 다음 수다.
- **위험도가 대체로 0 아니면 100이다.** 다만 어려운 입력을 주면 중간값이 나온다
  (51.0 / 43.1 / 41.1). 검증셋에 애매한 통화가 없었던 것이지 모델 결함은 아니다.
- **다채널 융합은 앱 몫이다.** 통화 위험도를 로그오즈로 바꿔 들고 있다가 문자 신호가 나오면
  더하는 방식 — `Voice-Detection/docs/METHOD.md` 6절.
- **`/transcribe`는 macOS에 묶여 있다.** `afconvert` 의존 하나 때문이다.

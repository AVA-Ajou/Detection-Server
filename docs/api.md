# API

## `GET /health`

적재 상태와 각 어댑터의 계약을 돌려준다. 배포 확인과 회귀 검증의 첫 단계.

```json
{"device": "mps",
 "tasks": {"voice": {"temperature": 1.37,
                     "calibrated": true,
                     "max_text_tokens": 1024,
                     "base_model": "google/gemma-3-4b-it"}}}
```

| 필드 | 뜻 |
|---|---|
| `device` | `cuda`(4bit) / `mps`(bf16) / `cpu`(fp32) |
| `calibrated` | `false`면 `calibration.json`이 없어 보정 없이 계산 중이다 |
| `base_model` | 계약이 지정한 모델. `BASE_MODEL` 환경변수로 덮어썼다면 실제와 다를 수 있다 |

모델을 아직 올리는 중이면 `503`.

## `POST /analyze`

```json
{"text": "여보세요 서울중앙지검…", "task": "voice", "reason": false}
```

| 필드 | 기본값 | |
|---|---|---|
| `text` | (필수) | 통화 전사본. 계약의 `max_text_tokens`를 넘으면 **앞부분만** 남기고 자른다 |
| `task` | `"voice"` | `adapters/` 아래 폴더 이름 |
| `reason` | `false` | 근거 문장까지 받을지 |

```json
{"task": "voice", "risk": 93.2, "probability": 0.932,
 "stage": 3, "stage_label": "정보·계좌",
 "reason": "\"안전계좌로 옮기셔야\"라며 이체를 준비시키고 있습니다.",
 "elapsed_ms": 12030}
```

| 필드 | |
|---|---|
| `risk` | 0~100. `probability × 100`을 반올림한 값이며 **화면에 그대로 쓰라고 있는 숫자**다 |
| `probability` | 온도 보정을 거친 `P("예")` |
| `stage` | 진행 단계 1~4. `reason: true`이고 위험도가 50 이상일 때만. **아래 경고 참고** |
| `stage_label` | 단계 이름 |
| `reason` | `reason: true`일 때만. 원문을 인용한 판정 근거 |
| `elapsed_ms` | 서버 측 소요. 락 대기 시간이 포함된다 |

### 진행 단계

위험도와 **다른 값**이다. 위험도가 "피싱이 맞나"라면 단계는 "지금 어디까지 왔나"다.

| | | |
|---|---|---|
| 1 | 접촉·사칭 | 기관이나 사람을 사칭했지만 아직 요구가 없다 |
| 2 | 압박·유인 | 범죄 연루·구속으로 겁을 주거나, 대출·환급을 미끼로 건다 |
| 3 | 정보·계좌 | 개인정보나 계좌번호를 요구하거나 불러준다. 앱 설치·원격 유도 |
| 4 | 이체 지시 | 송금·현금 전달·수거책 방문을 실제로 지시한다 |

> ⚠️ **현재 이 값은 신뢰할 수 없다.** 원본 Gemma가 생성으로 뱉는 값이라 거의 전부 3으로
> 쏠린다. 단계 경계를 아는 텍스트로 잰 결과 1/3/3/3이 나왔다(기대 1/2/3/4). 근거 인용은
> 맞게 하므로 독해가 아니라 분류가 안 되는 것이다. 자세한 내용과 다음 수(규칙 기반 전환)는
> `../Voice-Detection/docs/RESULTS.md`.

`stage`가 `null`인 경우 — 위험도가 50 미만이거나(정상 판정), 모델이 형식을 어겨 숫자를
못 읽었을 때. 후자에도 `reason`은 살린다.

### `reason`을 기본으로 켜지 말 것

| | 소요 | 하는 일 |
|---|---|---|
| 위험도 | 약 0.3초 | 순전파 한 번. 정답 자리 로짓을 읽는다 |
| 근거 | 15초 이상 | 어댑터를 끄고 256토큰 생성 |

모든 요청에 근거를 붙이면 정상 통화에도 15초를 쓰고 경보가 그만큼 늦어진다. 앱은 위험도를
먼저 받아 경보 여부를 정하고, **위험할 때만** 두 번째 요청을 보낸다.

에뮬레이터가 GPU를 함께 쓰면 근거 생성이 60초를 넘기기도 한다. 호출부는 근거 실패를 잡아
위험도만 쓰도록 만들 것 — 부가 기능이 판정을 막으면 안 된다.

### 오류

| 코드 | |
|---|---|
| `404` | `task`에 해당하는 어댑터가 없다. `/health`로 목록 확인 |
| `422` | `text`가 비었다 |
| `503` | 모델 적재 중 |

## 계약 파일 — `adapters/<task>/prompt.json`

서버는 이 파일만 보고 프롬프트를 재구성한다. **학습 저장소의 코드를 참조하지 않는다.**

```json
{"base_model": "google/gemma-3-4b-it",
 "instruction": "위 통화가 보이스피싱인지 판단하라. 예 또는 아니오로만 답하라.",
 "max_text_tokens": 1024,
 "yes": "예",
 "no": "아니오"}
```

재구성 순서다. **한 단계라도 어긋나면 에러 없이 확률만 틀어진다.**

```
① text를 토큰화해 앞 max_text_tokens 개만 남기고 다시 문자열로 decode
② f"{자른 텍스트}\n\n{instruction}" 를 user 메시지로
③ 토크나이저의 chat template 적용 (add_generation_prompt=True, tokenize=False)
④ 그 문자열을 add_special_tokens=False 로 인코딩   ← 템플릿이 <bos>를 이미 넣는다
⑤ 마지막 위치 로짓에서 yes/no 의 **첫 토큰** id 둘만 뽑아 softmax
```

`yes`/`no`는 토큰 수가 달라도 된다(`아니오`는 3토큰). 첫 토큰 id만 서로 다르면 된다.
같으면 적재 시점에 `SystemExit`으로 죽인다.

## 어댑터 추가

폴더 이름이 곧 `task` 값이 된다.

```bash
cp -R ../Voice-Detection/adapter adapters/sms     # checkpoint-* 는 제외
```

폴더에 있어야 하는 것.

| 파일 | 없으면 |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | 적재 실패 |
| `prompt.json` | 그 폴더를 어댑터로 인식하지 않는다 |
| `calibration.json` | 보정 없이 동작 (`/health`의 `calibrated: false`) |
| 토크나이저 (`tokenizer.json`, `chat_template.jinja` 등) | 적재 실패 |

학습이 끝난 어댑터에 계약만 붙이려면 학습 저장소에서:

```bash
python3 common.py --write-contract adapter/
```

여러 어댑터를 올려도 베이스 모델은 한 번만 적재된다. 요청마다 `set_adapter(task)`로 전환한다.

```
Gemma 3 4B   8.00GB     ← 한 번만
voice        0.125GB
sms          0.125GB
```

## 환경 변수

| | |
|---|---|
| `BASE_MODEL` | 계약의 `base_model`을 덮어쓴다. **같은 가중치의 미러에만 쓸 것** |
| `DEVICE=cpu` | GPU를 무시하고 CPU로 강제. 디버깅용 |

`google/gemma-3-4b-it`은 승인이 필요한 저장소다. 토큰이 없으면 `huggingface-cli login`을
하거나 `BASE_MODEL=unsloth/gemma-3-4b-it` 같은 미러를 쓴다. 미러를 썼다면 **반드시 알려진
값으로 대조할 것** — `../Voice-Detection/docs/RESULTS.md`에 기준선이 있다.

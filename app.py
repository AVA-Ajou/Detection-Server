#!/usr/bin/env python3
"""탐지 서버. 앱이 전사본을 보내면 위험도를 돌려준다.

    uvicorn app:app --host 0.0.0.0 --port 8000

`--host 0.0.0.0` 이어야 같은 와이파이의 폰에서 닿는다. 기본값(127.0.0.1)이면 맥 안에서만
보인다. 모델 적재에 수십 초가 걸리므로 첫 요청 전에 로그로 준비 완료를 확인할 것.
"""

import json as _json
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src import signals as _signals
from src.engine import Engine

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# signals.py의 ASSESS_THRESHOLD와 맞춘다 — 이 아래로는 단계를 계산하지 않는다.
_SMS_ASSESS_THRESHOLD = 20.0

_SMS_PROMPT = """\
다음 메시지(SMS 또는 카카오톡)가 보이스피싱·스미싱 금융사기인지 판단해라.

메시지:
{text}

JSON 하나만 출력해라:
{{"risk": <0~100 정수>}}

0 = 완전 정상, 100 = 확실한 피싱."""

engine = None


@asynccontextmanager
async def lifespan(_):
    # 요청마다 올리면 8GB를 매번 읽는다. 프로세스 수명 동안 한 번만 올린다.
    global engine
    # BASE_MODEL 은 계약(prompt.json)의 base_model 을 덮어쓴다. 같은 가중치의 미러를 쓸 때만
    # 건드릴 것 — 다른 모델을 넣으면 어댑터가 엉뚱한 곳에 붙어 확률이 조용히 망가진다.
    engine = Engine(base_model=os.environ.get("BASE_MODEL"),
                    device=os.environ.get("DEVICE"))
    print("준비 완료 — POST /analyze")
    yield


app = FastAPI(title="AVA Detection Server", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    task: str = "voice"
    # 근거 생성은 256토큰을 뽑아야 해서 위험도보다 수십 배 느리다. 앱이 위험도를 먼저 받고
    # 사용자가 "왜?"를 눌렀을 때 따로 요청하는 흐름을 기본으로 둔다.
    reason: bool = False


@app.get("/health")
def health():
    if engine is None:
        raise HTTPException(503, "모델을 아직 올리는 중입니다.")
    return engine.status()


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if req.task == "sms":
        return _analyze_sms(req.text)
    if engine is None:
        raise HTTPException(503, "모델을 아직 올리는 중입니다.")
    try:
        result = engine.analyze(req.task, req.text, req.reason)
    except KeyError as e:
        raise HTTPException(404, str(e))
    _log(req.text, result)
    return result


def _analyze_sms(text: str) -> dict:
    """SMS·카카오톡 텍스트를 Claude API로 판정한다.

    음성 모델(Gemma LoRA)은 통화 전사본에 특화되어 있어 단문 문자 채널에 쓰면
    보정 보장이 깨진다. SMS는 Claude에 위임하고 응답 포맷만 voice와 동일하게 맞춘다.
    단계 판정은 signals.py 정규식을 그대로 쓴다 — 이체·계좌·링크 패턴은 채널 무관하게 작동한다.
    """
    if not _ANTHROPIC_AVAILABLE:
        raise HTTPException(503, "anthropic 패키지가 없습니다. pip install anthropic")
    if not _ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")

    started = time.perf_counter()

    client = _anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": _SMS_PROMPT.format(text=text)}],
    )

    raw = message.content[0].text.strip()
    try:
        risk = float(max(0.0, min(100.0, _json.loads(raw).get("risk", 0))))
    except (ValueError, KeyError, _json.JSONDecodeError):
        raise HTTPException(500, f"Claude 응답 파싱 실패: {raw[:120]}")

    # 위험도가 문턱 위일 때만 단계를 계산한다. 정상 메시지에 단계를 붙이면
    # "사기가 진행 중"이라는 잘못된 신호가 된다.
    if risk >= _SMS_ASSESS_THRESHOLD:
        stage_num, evidence = _signals.stage_of(text)
    else:
        stage_num, evidence = None, []

    result = {
        "task": "sms",
        "risk": risk,
        "stage": stage_num,
        "stage_label": _signals.label(stage_num),
        "stage_evidence": [q for _, q in evidence],
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }
    _log(text, result)
    return result


def _log(text, result):
    """판정을 서버 콘솔에 한 줄로 남긴다.

    앱 화면에서 위험도 숫자를 뺐기 때문에(값이 0 아니면 100이라 정보가 없다) 이 로그가
    **점수를 볼 수 있는 유일한 자리**가 됐다. 정상 통화가 59.8 을 받는 것 같은 경계선은
    숫자로만 보인다. `uvicorn` 을 띄운 터미널에 그대로 찍힌다.
    """
    stage = result.get("stage")
    label = f"{stage}단계 {result.get('stage_label') or ''}".strip() if stage else "단계 없음"
    head = " ".join(text.split())[:46]
    print(f"  위험도 {result['risk']:5.1f}  {label:<12} {result['elapsed_ms']:>5}ms  {head}…",
          flush=True)


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...), task: str = "voice"):
    """통화 음성을 글로 옮긴다. 판정은 하지 않는다 — 그건 /analyze 몫이다.

    앱이 보내는 것은 대개 `.m4a`(AAC)인데 librosa 가 그 형식을 못 읽는다. macOS 에 기본으로
    있는 `afconvert` 로 16kHz 모노 wav 를 만들어 넘긴다. 다른 OS 에 올릴 때는 이 부분을
    ffmpeg 등으로 바꿔야 한다.
    """
    if engine is None:
        raise HTTPException(503, "모델을 아직 올리는 중입니다.")

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / (file.filename or "audio")
        source.write_bytes(file.file.read())
        if source.stat().st_size == 0:
            raise HTTPException(422, "빈 파일입니다.")

        started = time.perf_counter()
        try:
            audio = _to_wav(source)
        except RuntimeError as e:
            raise HTTPException(415, str(e))

        try:
            text = engine.transcribe(task, audio)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except RuntimeError as e:
            raise HTTPException(501, str(e))

    return {"text": text, "elapsed_ms": round((time.perf_counter() - started) * 1000)}


def _to_wav(source: Path) -> Path:
    """16kHz 모노 wav 로 바꾼다. 이미 wav 면 그대로 쓴다."""
    if source.suffix.lower() == ".wav":
        return source
    target = source.with_suffix(".wav")
    result = subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(source), str(target)],
        capture_output=True,
    )
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(
            f"'{source.suffix}' 를 wav 로 바꾸지 못했습니다: "
            f"{result.stderr.decode(errors='replace')[:200]}"
        )
    return target

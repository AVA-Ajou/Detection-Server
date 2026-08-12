#!/usr/bin/env python3
"""탐지 서버. 앱이 전사본을 보내면 위험도를 돌려준다.

    uvicorn app:app --host 0.0.0.0 --port 8000

`--host 0.0.0.0` 이어야 같은 와이파이의 폰에서 닿는다. 기본값(127.0.0.1)이면 맥 안에서만
보인다. 모델 적재에 수십 초가 걸리므로 첫 요청 전에 로그로 준비 완료를 확인할 것.
"""

import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.engine import Engine

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
    if engine is None:
        raise HTTPException(503, "모델을 아직 올리는 중입니다.")
    try:
        result = engine.analyze(req.task, req.text, req.reason)
    except KeyError as e:
        raise HTTPException(404, str(e))
    _log(req.text, result)
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

#!/usr/bin/env python3
"""탐지 서버. 앱이 전사본을 보내면 위험도를 돌려준다.

    uvicorn app:app --host 0.0.0.0 --port 8000

`--host 0.0.0.0` 이어야 같은 와이파이의 폰에서 닿는다. 기본값(127.0.0.1)이면 맥 안에서만
보인다. 모델 적재에 수십 초가 걸리므로 첫 요청 전에 로그로 준비 완료를 확인할 것.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine import Engine

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
        return engine.analyze(req.task, req.text, req.reason)
    except KeyError as e:
        raise HTTPException(404, str(e))

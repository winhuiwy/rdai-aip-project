import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, when the container/process starts — not per request.
    model.load()
    yield


app = FastAPI(title="Simple LLM Server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(200, ge=1, le=512)


class GenerateResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model.model is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if model.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    try:
        text = model.generate(req.prompt, req.max_new_tokens)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Generation failed") from exc
    return GenerateResponse(response=text)

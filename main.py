import asyncio
from contextlib import asynccontextmanager
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from batching_service import BatchingService
from inference_service import InferenceService
from mock_backend import MockBackend

mock_backend = MockBackend()
inference_service = InferenceService(mock_backend, timeout_seconds=0.5)
batching_service = BatchingService(
    mock_backend,
    max_batch_size=4,
    batch_timeout_seconds=0.01,
)


@asynccontextmanager
async def lifespan(app):
    await batching_service.start()
    try:
        yield
    finally:
        await batching_service.stop()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request, call_next):
    request.state.request_id = uuid.uuid4().hex

    response = await call_next(request)

    response.headers["X-Request-ID"] = request.state.request_id
    return response

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(..., min_length=1)

@app.post("/v1/chat/completions")
async def create_chat_completion(
    http_request: Request,
    request: ChatCompletionRequest,
):
    request_id = http_request.state.request_id

    try:
        backend_response = await asyncio.wait_for(
            batching_service.submit(request),
            timeout=inference_service.timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Inference request timed out.",
            headers={"X-Request-ID": request_id},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
            headers={"X-Request-ID": request_id},
        ) from exc

    return {
        "request_id": request_id,
        "latency_ms": backend_response["latency_ms"],
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": backend_response["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": backend_response["role"],
                    "content": backend_response["content"],
                },
                "finish_reason": "stop",
            }
        ],
    }
@app.get("/health")
def health_check():
    return {"status": "ok"}
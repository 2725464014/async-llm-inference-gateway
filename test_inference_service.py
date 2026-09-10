import asyncio

import pytest

from main import ChatCompletionRequest
from inference_service import InferenceService

class FakeBackend:
    def __init__(self):
        self.called_with = None

    async def generate(self, request):
        self.called_with = request
        return {
            "model": request.model,
            "role": "assistant",
            "content": "response from backend",
        }

@pytest.mark.asyncio
async def test_service_calls_backend():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            {"role": "user", "content": "hello"},
        ],
    )
    backend = FakeBackend()
    service = InferenceService(backend, timeout_seconds=1.0)

    response = await service.generate(request)

    assert response["content"] == "response from backend"
    assert backend.called_with is request

class SlowFakeBackend:
    async def generate(self, request):
        await asyncio.sleep(0.02)
        return {
            "model": request.model,
            "role": "assistant",
            "content": "slow response",
        }

@pytest.mark.asyncio
async def test_service_reports_backend_latency():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            {"role": "user", "content": "hello"},
        ],
    )
    backend = SlowFakeBackend()
    service = InferenceService(backend, timeout_seconds=1.0)

    response = await service.generate(request)

    assert response["content"] == "slow response"
    assert response["latency_ms"] >= 20

class HangingFakeBackend:
    async def generate(self, request):
        await asyncio.sleep(1)

@pytest.mark.asyncio
async def test_service_times_out_slow_backend():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            {"role": "user", "content": "hello"},
        ],
    )
    backend = HangingFakeBackend()
    service = InferenceService(backend, timeout_seconds=0.01)

    with pytest.raises(TimeoutError):
        await service.generate(request)
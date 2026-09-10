import asyncio

import pytest

from batching_service import BatchingService
from main import ChatCompletionRequest


class RecordingBackend:
    def __init__(self):
        self.batch_sizes = []

    async def generate_batch(self, requests):
        self.batch_sizes.append(len(requests))
        return [
            {
                "model": request.model,
                "role": "assistant",
                "content": f"response: {request.messages[-1].content}",
            }
            for request in requests
        ]


def make_request(content):
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": content}],
    )


@pytest.mark.asyncio
async def test_single_request_passes_through_queue():
    backend = RecordingBackend()
    service = BatchingService(backend, max_batch_size=4, batch_timeout_seconds=0.05)

    await service.start()
    try:
        response = await service.submit(make_request("hello"))
    finally:
        await service.stop()

    assert response["content"] == "response: hello"
    assert backend.batch_sizes == [1]


@pytest.mark.asyncio
async def test_concurrent_requests_are_processed_as_one_batch():
    backend = RecordingBackend()
    service = BatchingService(backend, max_batch_size=4, batch_timeout_seconds=0.05)

    await service.start()
    try:
        responses = await asyncio.gather(
            service.submit(make_request("first")),
            service.submit(make_request("second")),
            service.submit(make_request("third")),
        )
    finally:
        await service.stop()

    assert backend.batch_sizes == [3]
    assert [response["content"] for response in responses] == [
        "response: first",
        "response: second",
        "response: third",
    ]


@pytest.mark.asyncio
async def test_batch_timeout_processes_incomplete_batch():
    backend = RecordingBackend()
    service = BatchingService(backend, max_batch_size=8, batch_timeout_seconds=0.01)

    await service.start()
    try:
        response = await service.submit(make_request("timeout-triggered"))
    finally:
        await service.stop()

    assert response["content"] == "response: timeout-triggered"
    assert backend.batch_sizes == [1]

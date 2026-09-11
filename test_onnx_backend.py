import pytest

from main import ChatCompletionRequest
from onnx_backend import OnnxBackend


def make_request(content):
    return ChatCompletionRequest(
        model="onnx-test",
        messages=[{"role": "user", "content": content}],
    )


@pytest.mark.asyncio
async def test_onnx_backend_runs_one_batched_inference():
    backend = OnnxBackend()

    responses = await backend.generate_batch(
        [make_request("first"), make_request("second")]
    )

    assert len(responses) == 2
    assert all(response["role"] == "assistant" for response in responses)
    assert all(response["model"] == "onnx-test" for response in responses)
    assert all(response["content"].startswith("ONNX score: ") for response in responses)
    assert backend.batch_calls == 1
    assert backend.batch_shapes == [2]


@pytest.mark.asyncio
async def test_onnx_single_request_matches_batch_shape():
    backend = OnnxBackend()

    response = await backend.generate(make_request("one"))

    assert response["content"].startswith("ONNX score: ")

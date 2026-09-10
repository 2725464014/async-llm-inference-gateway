import pytest

from main import ChatCompletionRequest
from mock_backend import MockBackend

@pytest.mark.asyncio
async def test_backend_preserves_model_name():
    request = ChatCompletionRequest(
        model="my-test-model",
        messages=[
            {"role": "user", "content": "hello"},
        ],
    )

    backend = MockBackend()

    response = await backend.generate(request)

    assert response["model"] == "my-test-model"

@pytest.mark.asyncio
async def test_backend_uses_last_user_message():
    request = ChatCompletionRequest(
        model="mock-llm",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "last"},
        ],
    )

    backend = MockBackend()

    response = await backend.generate(request)

    assert response["content"] == "Mock response: last"

@pytest.mark.asyncio
async def test_backend_returns_assistant_role():
    request = ChatCompletionRequest(
        model="mock-llm",
        messages=[
            {"role": "user", "content": "hello"},
        ],
    )

    backend = MockBackend()

    response = await backend.generate(request)

    assert response["role"] == "assistant"
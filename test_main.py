import main
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_request_uses_last_user_message():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "last"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "mock-llm"
    assert data["choices"][0]["message"]["content"] == "Mock response: last"


def test_missing_messages_returns_422():
    response = client.post("/v1/chat/completions", json={"model": "mock-llm"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "messages"]


def test_empty_messages_returns_422():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-llm", "messages": []},
    )

    assert response.status_code == 422
    assert "messages" in response.json()["detail"][0]["loc"]


def test_invalid_role_returns_422():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "invalid", "content": "hello"}],
        },
    )

    assert response.status_code == 422
    assert "role" in response.json()["detail"][0]["loc"]

def test_empty_string_content_returns_422():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": ""}],
        },
    )

    assert response.status_code == 422
    assert "content" in response.json()["detail"][0]["loc"]

def test_non_string_content_returns_422():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": 123}],
        },
    )

    assert response.status_code == 422
    assert "content" in response.json()["detail"][0]["loc"]


def test_request_without_user_message_returns_400():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "assistant", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "No user message found in the request."}


def test_model_name_is_preserved():
    model_name = "my-custom-test-model"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == model_name

def test_endpoint_uses_mock_backend(monkeypatch):
    async def fake_generate(request):
        return {
            "model": request.model,
            "role": "assistant",
            "content": "response from fake backend",
        }

    monkeypatch.setattr(main.mock_backend, "generate", fake_generate)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == (
        "response from fake backend"
    )

def test_endpoint_uses_batching_service(monkeypatch):
    async def fake_submit(request):
        return {
            "model": request.model,
            "role": "assistant",
            "content": "response from service",
            "latency_ms": 1.25,
            "queue_wait_ms": 0.1,
            "backend_execution_ms": 1.0,
            "batch_size": 1,
        }

    monkeypatch.setattr(
        main.batching_service,
        "submit",
        fake_submit,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"]
        == "response from service"
    )

def test_response_contains_request_id():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["request_id"], str)
    assert data["request_id"]

def test_each_request_gets_unique_request_id():
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
    }

    first = client.post("/v1/chat/completions", json=payload)
    second = client.post("/v1/chat/completions", json=payload)

    assert first.json()["request_id"] != second.json()["request_id"]

def test_response_contains_latency_ms():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["latency_ms"], float)
    assert data["latency_ms"] >= 0

def test_backend_timeout_returns_504(monkeypatch):
    async def timeout_submit(request):
        raise TimeoutError

    monkeypatch.setattr(
        main.batching_service,
        "submit",
        timeout_submit,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 504
    assert response.json() == {
        "detail": "Inference request timed out."
    }

def test_timeout_response_contains_request_id(monkeypatch):
    async def timeout_submit(request):
        raise TimeoutError

    monkeypatch.setattr(
        main.batching_service,
        "submit",
        timeout_submit,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 504
    assert response.headers["X-Request-ID"]
    assert len(response.headers["X-Request-ID"]) == 32

def test_bad_request_response_contains_request_id():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "assistant", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "No user message found in the request."
    }
    assert response.headers["X-Request-ID"]
    assert len(response.headers["X-Request-ID"]) == 32

def test_validation_error_response_contains_request_id():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [],
        },
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"]
    assert len(response.headers["X-Request-ID"]) == 32

def test_success_request_id_matches_response_header():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["request_id"] == response.headers["X-Request-ID"]
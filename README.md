# Async LLM Inference Gateway

A learning-focused inference gateway that demonstrates asynchronous request scheduling, dynamic batching, per-request result delivery, timeout handling, request tracing, and reproducible performance experiments.

> This project currently uses a controlled mock backend. It validates gateway scheduling behavior, not real LLM or GPU inference performance.

## Architecture

```text
HTTP client
    |
    v
FastAPI + Pydantic validation
    |
    v
BatchingService.submit(request)
    |
    v
asyncio.Queue
    |
    v
Background batch worker
    |  max_batch_size / batch_timeout
    v
MockBackend.generate_batch(requests)
    |
    v
per-request Future -> HTTP response
```

Each queued item contains the request, a per-request `Future`, and its enqueue timestamp. The worker collects requests until the batch reaches `max_batch_size` or `batch_timeout_seconds` expires, invokes the backend once, and resolves each request's corresponding Future in input order.

## Implemented Features

- OpenAI-style `POST /v1/chat/completions` endpoint
- Nested Pydantic request validation
- Request IDs shared between the JSON body and `X-Request-ID` response header
- HTTP 400, 422, and 504 behavior
- Configurable request timeout
- `asyncio.Queue` request scheduling
- Per-request `Future` result delivery
- Configurable maximum batch size and batch timeout
- Queue-wait, end-to-end latency, and batch-size measurements
- Direct-versus-batched benchmark with controlled concurrency
- Automated tests for API, backend, timeout, queue, and batching behavior

## Run Locally

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Health check:

```powershell
curl http://127.0.0.1:8001/health
```

Example request:

```powershell
curl -X POST http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"mock-llm","messages":[{"role":"user","content":"hello"}]}'
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Verified baseline:

```text
28 passed, 1 warning
```

The remaining warning is a Starlette/httpx TestClient deprecation warning and does not represent a failed test.

## Benchmark

Example:

```powershell
.\.venv\Scripts\python.exe benchmark.py --mode both --requests 64 --concurrency 4 --max-batch-size 4 --batch-timeout-ms 5
```

The benchmark reports:

- throughput in requests per second
- error rate
- mean, P50, P95, and P99 latency
- mean and P95 queue wait
- observed batch-size distribution

Raw results are stored in [`benchmark_results.json`](benchmark_results.json).

## Controlled Experiment

Configuration:

- 64 requests per mode
- concurrency: 1, 4, and 16
- maximum batch size: 4
- batch timeout: 5 ms
- controlled direct backend delay: 20 ms per request
- controlled batched backend delay: 10 ms per batch

| Concurrency | Mode | Throughput (req/s) | Mean latency (ms) | P95 (ms) | Queue wait mean (ms) | Observed batch size |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Direct | 32.22 | 31.02 | 32.09 | 0.00 | N/A |
| 1 | Batched | 64.74 | 15.43 | 16.05 | 0.11 | 1 |
| 4 | Direct | 129.16 | 30.87 | 31.97 | 0.00 | N/A |
| 4 | Batched | 257.10 | 15.47 | 16.33 | 0.15 | 4 |
| 16 | Direct | 528.76 | 30.07 | 31.65 | 0.00 | N/A |
| 16 | Batched | 257.48 | 56.20 | 62.93 | 40.77 | 4 |

## Interpretation

At concurrency 4, requests consistently formed batches of four and the controlled batch-efficient backend produced higher throughput and lower latency than the direct path. At concurrency 16, a single batch worker became the bottleneck: batches remained full, but queue wait increased to about 41 ms on average and P95 latency rose to about 63 ms.

This demonstrates an important serving trade-off: higher concurrency can improve batch utilization, but a single worker can saturate and increase queueing delay and tail latency.

The concurrency-1 result should not be interpreted as real model acceleration. The controlled backend intentionally uses different fixed delays for direct and batched calls to make scheduling behavior measurable. Real performance conclusions require a real model runtime and hardware-backed experiments.

## Current Limitations

- The backend returns mock text rather than running an LLM.
- `generate_batch()` does not perform tensor-level GPU batching.
- The benchmark models backend costs with controlled delays.
- There is one in-process queue and one batch worker.
- There is no load shedding, retry policy, metrics exporter, or distributed worker pool.

## Next Milestone

Integrate a lightweight real inference backend and repeat the experiment matrix so the project can report measured model-serving results rather than simulated backend behavior.

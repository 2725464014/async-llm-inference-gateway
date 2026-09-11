# Async LLM Inference Gateway

An asynchronous inference gateway prototype demonstrating request scheduling, dynamic batching, per-request Futures, timeout handling, request tracing, and reproducible CPU inference experiments.

> The gateway supports a real CPU ONNX Runtime backend. It is not yet a generative LLM server and does not claim GPU performance.

## Architecture

```text
Client -> FastAPI -> BatchingService -> asyncio.Queue -> batch worker
       <- HTTP response <- per-request Future <- backend.generate_batch()
```

The worker collects requests until `max_batch_size` is reached or `batch_timeout_seconds` expires. Each request keeps its own Future, so batched responses are returned to the correct caller in input order.

## Features

- OpenAI-style `POST /v1/chat/completions`
- Pydantic validation and HTTP 400/422/504 handling
- Request ID in both JSON and `X-Request-ID`
- `asyncio.Queue` and background batch worker
- Configurable maximum batch size and batch timeout
- Queue-wait, backend execution, end-to-end latency, and batch-size measurements
- Controlled benchmark plus real CPU ONNX Runtime benchmark
- P50/P95/P99 latency, throughput, and error rate
- Graceful shutdown that unblocks pending requests

## Run Locally

Install dependencies in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the default mock backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Run the real CPU ONNX backend:

```powershell
$env:INFERENCE_BACKEND="onnx"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

The tiny ONNX model is generated deterministically on first use under `models/` and is ignored by Git because it is reproducible from `onnx_model.py`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Verified baseline:

```text
31 passed, 1 warning
```

The remaining warning is a Starlette/httpx TestClient deprecation warning and does not represent a failed test.

## Benchmarks

Controlled scheduling benchmark:

```powershell
.\.venv\Scripts\python.exe benchmark.py --backend controlled --mode both --requests 64 --concurrency 4 --max-batch-size 4 --batch-timeout-ms 5
```

Real CPU ONNX benchmark:

```powershell
.\.venv\Scripts\python.exe benchmark.py --backend onnx --mode both --requests 64 --concurrency 4 --max-batch-size 4 --batch-timeout-ms 5
```

The benchmark reports throughput, error rate, mean/P50/P95/P99 latency, queue wait, backend execution time, and observed batch-size distribution.

## Real ONNX Runtime Experiment

Configuration: 64 requests per mode, maximum batch size 4, 5 ms batch timeout, CPUExecutionProvider.

| Concurrency | Mode | Throughput req/s | Mean latency ms | P95 ms | Queue wait mean ms | Backend execution mean ms | Batch distribution |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Direct | 3,313.47 | 0.286 | 0.229 | 0.000 | 0.286 | N/A |
| 1 | Batched | 4,539.72 | 0.210 | 0.300 | 0.056 | 0.137 | 1:64 |
| 4 | Direct | 6,188.84 | 0.381 | 0.546 | 0.000 | 0.381 | N/A |
| 4 | Batched | 16,478.28 | 0.211 | 0.346 | 0.060 | 0.133 | 4:64 |
| 16 | Direct | 6,336.13 | 0.977 | 1.482 | 0.000 | 0.977 | N/A |
| 16 | Batched | 14,159.29 | 0.945 | 1.637 | 0.754 | 0.165 | 4:60, 3:3, 1:1 |

Raw ONNX results are stored in [`onnx_benchmark_results.json`](onnx_benchmark_results.json). Earlier controlled-delay results are stored in [`benchmark_results.json`](benchmark_results.json).

## Interpretation

At concurrency 4, the ONNX batched path formed 64 full batches of four and achieved higher measured throughput than the direct path in this environment. The mean backend execution time per batch was about 0.133 ms, while the direct path measured about 0.381 ms per request. At concurrency 16, batching still improved throughput, but queue wait increased and P95 latency rose, showing the trade-off between utilization and tail latency.

These measurements are CPU results for a tiny deterministic MLP, not generative LLM results. They demonstrate real tensor batching through one ONNX Runtime `session.run()` call per batch. They should not be presented as GPU or production LLM serving performance.

## Current Limitations

- The ONNX model is a tiny deterministic text-feature MLP, not a generative LLM.
- There is one in-process queue and one batch worker.
- There is no load shedding, retry policy, metrics exporter, or distributed worker pool.
- The benchmark is CPU-only and hardware-specific.

## Next Milestone

Integrate a small real text model with a documented tokenizer and repeat the direct-versus-batched experiment with model-specific metrics.

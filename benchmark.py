import argparse
import asyncio
import json
import statistics
import time
from collections import Counter

from batching_service import BatchingService
from main import ChatCompletionRequest
from onnx_backend import OnnxBackend


class ControlledBenchmarkBackend:
    """Controlled backend for a fair direct-vs-batched comparison."""

    async def generate(self, request):
        await asyncio.sleep(0.020)
        return {
            "model": request.model,
            "role": "assistant",
            "content": f"response: {request.messages[-1].content}",
        }

    async def generate_batch(self, requests):
        await asyncio.sleep(0.010)
        return [
            {
                "model": request.model,
                "role": "assistant",
                "content": f"response: {request.messages[-1].content}",
            }
            for request in requests
        ]


def make_request(index: int) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mock-llm",
        messages=[
            {"role": "user", "content": f"benchmark request {index}"},
        ],
    )


def percentile(values, percentile_value):
    if not values:
        return 0.0

    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


async def run_direct(total_requests: int, concurrency: int, backend):
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async def send(index):
        nonlocal errors
        async with semaphore:
            started = time.perf_counter()
            try:
                await backend.generate(make_request(index))
            except Exception:
                errors += 1
                return
            latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(send(index) for index in range(total_requests)))
    elapsed = time.perf_counter() - started

    return summarize(
        mode="direct",
        total_requests=total_requests,
        completed=len(latencies),
        errors=errors,
        elapsed_seconds=elapsed,
        latencies=latencies,
        queue_waits=[],
        backend_executions=latencies,
        batch_sizes=[],
    )


async def run_batched(
    total_requests: int,
    concurrency: int,
    max_batch_size: int,
    batch_timeout_seconds: float,
    backend,
):
    service = BatchingService(
        backend,
        max_batch_size=max_batch_size,
        batch_timeout_seconds=batch_timeout_seconds,
    )
    await service.start()

    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    queue_waits = []
    backend_executions = []
    batch_sizes = []
    errors = 0

    async def send(index):
        nonlocal errors
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await service.submit(make_request(index))
            except Exception:
                errors += 1
                return

            latencies.append((time.perf_counter() - started) * 1000)
            queue_waits.append(response["queue_wait_ms"])
            backend_executions.append(response["backend_execution_ms"])
            batch_sizes.append(response["batch_size"])

    started = time.perf_counter()
    try:
        await asyncio.gather(*(send(index) for index in range(total_requests)))
    finally:
        await service.stop()
    elapsed = time.perf_counter() - started

    return summarize(
        mode="batched",
        total_requests=total_requests,
        completed=len(latencies),
        errors=errors,
        elapsed_seconds=elapsed,
        latencies=latencies,
        queue_waits=queue_waits,
        backend_executions=backend_executions,
        batch_sizes=batch_sizes,
    )


def summarize(
    mode,
    total_requests,
    completed,
    errors,
    elapsed_seconds,
    latencies,
    queue_waits,
    backend_executions,
    batch_sizes,
):
    return {
        "mode": mode,
        "total_requests": total_requests,
        "completed": completed,
        "errors": errors,
        "error_rate": errors / total_requests if total_requests else 0.0,
        "throughput_requests_per_second": (
            completed / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "queue_wait_ms": {
            "mean": statistics.fmean(queue_waits) if queue_waits else 0.0,
            "p95": percentile(queue_waits, 95),
        },
        "backend_execution_ms": {
            "mean": (
                statistics.fmean(backend_executions)
                if backend_executions
                else 0.0
            ),
            "p95": percentile(backend_executions, 95),
        },
        "batch_size_distribution": dict(Counter(batch_sizes)),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "batched", "both"], default="both")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-timeout-ms", type=float, default=5.0)
    parser.add_argument("--backend", choices=["controlled", "onnx"], default="controlled")
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")

    results = []
    backend_factory = (
        OnnxBackend if args.backend == "onnx" else ControlledBenchmarkBackend
    )
    if args.mode in {"direct", "both"}:
        result = await run_direct(
            args.requests,
            args.concurrency,
            backend_factory(),
        )
        result["backend"] = args.backend
        results.append(result)
    if args.mode in {"batched", "both"}:
        result = await run_batched(
            args.requests,
            args.concurrency,
            args.max_batch_size,
            args.batch_timeout_ms / 1000,
            backend_factory(),
        )
        result["backend"] = args.backend
        results.append(result)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

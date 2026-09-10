import asyncio
import time

class InferenceService:
    def __init__(self, backend, timeout_seconds):
        self.backend = backend
        self.timeout_seconds = timeout_seconds

    async def generate(self, request):
        start_time = time.perf_counter()

        response = await asyncio.wait_for(
            self.backend.generate(request),
            timeout=self.timeout_seconds,
        )

        elapsed_seconds = time.perf_counter() - start_time

        return {
            **response,
            "latency_ms": elapsed_seconds * 1000,
        }
import asyncio
import time
from dataclasses import dataclass


@dataclass
class QueueItem:
    request: object
    future: asyncio.Future
    enqueued_at: float


class BatchingService:
    def __init__(
        self,
        backend,
        max_batch_size: int,
        batch_timeout_seconds: float,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if batch_timeout_seconds <= 0:
            raise ValueError("batch_timeout_seconds must be positive")

        self.backend = backend
        self.max_batch_size = max_batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.queue = asyncio.Queue()
        self.worker_task = None

    async def start(self):
        if self.worker_task is None or self.worker_task.done():
            self.queue = asyncio.Queue()
            self.worker_task = asyncio.create_task(self._worker_loop())

    async def submit(self, request):
        if self.worker_task is None or self.worker_task.done():
            await self.start()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(
            QueueItem(
                request=request,
                future=future,
                enqueued_at=time.perf_counter(),
            )
        )
        return await future

    async def stop(self):
        if self.worker_task is None:
            return

        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass
        finally:
            self.worker_task = None

    async def _worker_loop(self):
        while True:
            first_item = await self.queue.get()
            batch = [first_item]

            deadline = asyncio.get_running_loop().time() + (
                self.batch_timeout_seconds
            )

            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break

                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    break

                batch.append(item)

            await self._process_batch(batch)

    async def _process_batch(self, batch):
        requests = [item.request for item in batch]
        batch_started_at = time.perf_counter()

        try:
            responses = await self.backend.generate_batch(requests)
            if len(responses) != len(batch):
                raise RuntimeError(
                    "Backend returned a different number of responses"
                )
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        completed_at = time.perf_counter()
        for item, response in zip(batch, responses):
            if not item.future.done():
                item.future.set_result(
                    {
                        **response,
                        "queue_wait_ms": (
                            batch_started_at - item.enqueued_at
                        ) * 1000,
                        "latency_ms": (
                            completed_at - item.enqueued_at
                        ) * 1000,
                        "batch_size": len(batch),
                    }
                )

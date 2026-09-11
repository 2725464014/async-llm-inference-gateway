import asyncio
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from onnx_model import FEATURE_DIM, ensure_model


class OnnxBackend:
    """Small real CPU inference backend using one ONNX Runtime batch call."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = ensure_model(
            Path(model_path) if model_path is not None else None
        )
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.batch_calls = 0
        self.batch_shapes = []

    @staticmethod
    def _last_user_content(request) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content
        raise ValueError("No user message found in the request.")

    @classmethod
    def _features(cls, request) -> np.ndarray:
        content = cls._last_user_content(request)
        digest = hashlib.sha256(content.encode("utf-8")).digest()
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        for index, value in enumerate(digest):
            features[index] = value / 255.0
        features[26] = min(len(content), 256) / 256.0
        features[27] = content.count(" ") / max(len(content), 1)
        return features

    def _run_session(self, features: np.ndarray) -> np.ndarray:
        return self.session.run(
            [self.output_name],
            {self.input_name: features},
        )[0]

    async def generate(self, request):
        return (await self.generate_batch([request]))[0]

    async def generate_batch(self, requests):
        if not requests:
            return []

        features = np.stack([self._features(request) for request in requests])
        self.batch_calls += 1
        self.batch_shapes.append(len(requests))
        scores = await asyncio.to_thread(self._run_session, features)

        return [
            {
                "model": request.model,
                "role": "assistant",
                "content": f"ONNX score: {float(score[0]):.6f}",
            }
            for request, score in zip(requests, scores)
        ]

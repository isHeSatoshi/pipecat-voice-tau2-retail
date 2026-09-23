"""Parakeet STT service (nemo_toolkit[asr]).

Wraps NVIDIA Parakeet via NeMo's ``ASRModel`` so the harness can transcribe
locally without going through a cloud STT API. Parakeet is a CTC/tdt
transducer model that supports streaming-style inference for short
utterances; we use it in non-streaming mode for the eval baseline because
the audio bus delivers small chunks.

This module imports ``nemo.collections.asr`` lazily so the harness still
imports cleanly when Parakeet is not installed (use ``--stt dummy``).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat_voice.interfaces import STTResult

_PARAKEET_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="parakeet-gpu"
)


@dataclass
class ParakeetSTT:
    """Parakeet STT backed by nemo_toolkit[asr].

    The model is loaded lazily on the first ``transcribe`` call so import
    time stays fast and tests can construct the object before any GPU work
    happens.
    """

    sample_rate_in: int = 16000
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    device: str = "cuda"
    _model: Any = field(default=None, init=False, repr=False)  # noqa: F821

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from nemo.collections.asr.models import ASRModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ParakeetSTT requires `nemo_toolkit[asr]`. Install with "
                "`uv pip install nemo_toolkit[asr]` or use `--stt dummy`."
            ) from e
        logger.info(f"Loading Parakeet model: {self.model_name}")
        self._model = ASRModel.from_pretrained(model_name=self.model_name)
        try:
            self._model = self._model.to(self.device)
        except Exception:
            logger.warning("Could not move Parakeet to {self.device}; staying on CPU.")
        self._model.eval()

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> STTResult:
        """Transcribe a chunk of 16-bit PCM mono audio at ``sample_rate`` Hz.

        Parakeet (NeMo) accepts raw numpy arrays or tensors, so we decode the
        PCM bytes ourselves rather than wrapping in a WAV. Inference runs in
        a thread pool so the event loop is not blocked.

        Near-silent or very short audio short-circuits to ``""`` without
        touching the GPU: the harness flushes fixed windows that are often
        pure silence.
        """
        if not pcm_bytes or len(pcm_bytes) < sample_rate * 2 * 0.1:
            return STTResult(text="", confidence=0.0, is_final=True)
        if _is_silence(pcm_bytes):
            return STTResult(text="", confidence=0.0, is_final=True)

        self._ensure_model()

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            _PARAKEET_EXECUTOR, self._infer, pcm_bytes, sample_rate
        )
        return STTResult(text=text.strip(), confidence=1.0, is_final=True)

    def _infer(self, pcm_bytes: bytes, sample_rate: int) -> str:
        import numpy as np

        # Decode 16-bit PCM mono bytes to float32 in [-1, 1].
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""
        try:
            output = self._model.transcribe([audio])
        except Exception as e:
            logger.warning(f"Parakeet inference failed: {e}")
            return ""
        if not output:
            return ""
        first = output[0]
        if hasattr(first, "text"):
            return first.text
        if isinstance(first, str):
            return first
        return str(first)


def _is_silence(pcm_bytes: bytes, *, threshold: int = 100) -> bool:
    """True when every 16-bit sample is below ``threshold`` (near-silence).

    Pure-Python scan over the buffer header: checks the peak over a stride
    so a multi-second window costs microseconds, not a numpy import.
    """
    import struct

    n = len(pcm_bytes) // 2
    if n == 0:
        return True
    stride = max(1, n // 2000)
    peak = 0
    for i in range(0, n, stride):
        (v,) = struct.unpack_from("<h", pcm_bytes, i * 2)
        a = v if v >= 0 else -v
        if a > peak:
            peak = a
            if peak >= threshold:
                return False
    return peak < threshold

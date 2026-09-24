"""Chatterbox TTS service.

Wraps the ``chatterbox-tts`` package so the harness can synthesise audio
locally without a cloud TTS API.

Chatterbox produces 24 kHz PCM by default. The bus is configured for 16 kHz
to match Parakeet's preferred input rate; we resample if needed using
``scipy.signal.resample_poly``.

Lazy import of ``chatterbox`` so the harness remains importable without the
heavy torchaudio dependency.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from loguru import logger

from pipecat_voice.interfaces import TTSChunk

# Chatterbox's compiled kernels (CUDA graphs) are bound to the thread/stream
# that first runs them: concurrent inference from two executor threads
# crashes with "Offset increment outside graph capture", and even serialised
# calls from different threads can replay graphs on the wrong stream. ALL
# GPU-touching Chatterbox work (load + inference) therefore runs on one
# dedicated thread via _CHATTERBOX_EXECUTOR.
_CHATTERBOX_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="chatterbox-gpu"
)


@dataclass
class ChatterboxTTS:
    """Chatterbox TTS backed by the ``chatterbox-tts`` package."""

    sample_rate_out: int = 16000
    voice: str = "default"
    device: str = "cuda"
    _model: Any = field(default=None, init=False, repr=False)  # noqa: F821

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from chatterbox.tts import ChatterboxTTS as _ChatterboxTTS  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ChatterboxTTS requires `chatterbox-tts`. Install with "
                "`uv pip install chatterbox-tts` or use `--tts dummy`."
            ) from e
        logger.info(f"Loading Chatterbox TTS model on {self.device}...")
        # `from_pretrained` requires a device string ("cuda", "cpu", "mps").
        # NOTE: must run on _CHATTERBOX_EXECUTOR (see module docstring); call
        # the async `ensure_loaded` from async code instead of calling this
        # directly from arbitrary threads.
        self._model = _ChatterboxTTS.from_pretrained(device=self.device)

    async def ensure_loaded(self) -> None:
        """Load model weights on the dedicated GPU thread (idempotent)."""
        if self._model is not None:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_CHATTERBOX_EXECUTOR, self._ensure_model)

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:
        if not text.strip():
            return
        await self.ensure_loaded()
        loop = asyncio.get_event_loop()
        self._synth_calls = getattr(self, "_synth_calls", 0) + 1
        self._synth_chars = getattr(self, "_synth_chars", 0) + len(text)
        try:
            wav = await loop.run_in_executor(_CHATTERBOX_EXECUTOR, self._infer, text)
        except Exception as e:
            # Transient inductor/CUDA-graphs failures ("Offset increment
            # outside graph capture") strike the first concurrent uses and
            # clear on their own; one retry after a short cooldown recovers
            # the turn instead of dropping it.
            logger.warning(f"chatterbox infer failed ({e}); retrying once")
            await asyncio.sleep(2.0)
            wav = await loop.run_in_executor(_CHATTERBOX_EXECUTOR, self._infer, text)
        logger.info(
            f"chatterbox synth #{self._synth_calls} chars={len(text)} "
            f"total_chars={self._synth_chars} text={text[:80]!r}"
        )
        # Chatterbox exposes its native sample rate via `self._model.sr`.
        sr_in = getattr(self._model, "sr", self.sample_rate_out)
        # Convert to 16-bit PCM mono, resampling if necessary.
        pcm_bytes = await loop.run_in_executor(
            None, self._to_pcm16, wav, sr_in, self.sample_rate_out
        )
        # Stream in 100 ms slices for parity with the dummy implementation.
        slice_bytes = self.sample_rate_out * 2 * 100 // 1000
        for i in range(0, len(pcm_bytes), slice_bytes):
            is_final = i + slice_bytes >= len(pcm_bytes)
            yield TTSChunk(
                pcm=pcm_bytes[i : i + slice_bytes],
                sample_rate=self.sample_rate_out,
                is_final=is_final,
            )

    def _infer(self, text: str):
        # Runs on _CHATTERBOX_EXECUTOR (single thread) — see module docstring.
        # Chatterbox's `generate` returns a single torch.Tensor of shape
        # (1, n). Newer builds may return a tuple (wav, sr); unwrap here so
        # `_to_pcm16` always sees the tensor.
        out = self._model.generate(text)
        if isinstance(out, (tuple, list)) and out:
            return out[0]
        return out

    @staticmethod
    def _to_pcm16(wav: Any, sr_in: int, sr_out: int) -> bytes:
        import numpy as np

        # wav is typically a torch.Tensor of shape (1, samples).
        if isinstance(wav, (tuple, list)) and wav:
            wav = wav[0]
        try:
            arr = wav.squeeze(0).cpu().numpy()
        except Exception:
            arr = wav
        arr = np.asarray(arr, dtype=np.float32).squeeze()
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if sr_in != sr_out:
            from scipy.signal import resample_poly

            arr = resample_poly(arr, sr_out, sr_in)
        arr = np.clip(arr, -1.0, 1.0)
        int16 = (arr * 32767.0).astype("int16")
        return int16.tobytes()

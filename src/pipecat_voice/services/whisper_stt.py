"""Whisper STT placeholder.

This is a thin stub so the harness can be invoked with ``--stt whisper``
without forcing a hard dependency on the OpenAI SDK. Real implementations
should subclass Pipecat's ``WhisperSTTService`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from pipecat_voice.interfaces import STTResult


@dataclass
class WhisperSTT:
    sample_rate_in: int = 16000

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> STTResult:
        raise NotImplementedError(
            "WhisperSTT stub: wire pipecat.services.whisper.stt.WhisperSTTService "
            "in runner._build_stt for --stt whisper."
        )

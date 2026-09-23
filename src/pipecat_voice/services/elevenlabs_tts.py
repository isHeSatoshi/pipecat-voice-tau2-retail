"""ElevenLabs TTS placeholder.

Thin stub so the harness can be invoked with ``--tts elevenlabs`` without
forcing a hard dependency on the ElevenLabs SDK at import time. Real
implementations should subclass Pipecat's ``ElevenLabsTTSService``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from pipecat_voice.interfaces import TTSChunk


@dataclass
class ElevenLabsTTS:
    sample_rate_out: int = 16000
    voice: str = "default"

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:
        raise NotImplementedError(
            "ElevenLabsTTS stub: wire pipecat.services.elevenlabs.tts.ElevenLabsTTSService "
            "in runner._build_tts for --tts elevenlabs."
        )
        yield  # pragma: no cover -- unreachable

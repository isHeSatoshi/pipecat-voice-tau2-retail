"""Dummy STT, TTS, and LLM implementations.

These satisfy the protocols defined in :mod:`pipecat_voice.interfaces` but
do not perform any real inference. They are used for:

- CI smoke tests that must not require GPU, audio backends, or network keys.
- Local development where a developer wants to verify the loop closes
  before plugging in real Parakeet / Chatterbox / MiniMax.

Behaviour
---------

- :class:`DummySTT` echoes the most recent text it was ``prime``-d with. This
  lets a test drive the agent pipeline with controlled user utterances
  without round-tripping through STT.

- :class:`DummyTTS` emits silence (zeroed PCM) at the configured sample
  rate. The bytes are valid PCM so downstream STT will see silence.

- :class:`DummyLLM` walks through a queued script of ``LLMResponse``
  instances and falls back to a trivial deterministic reply. Useful for
  asserting protocol contracts in tests.

The real implementations (Parakeet, Chatterbox, MiniMax) are imported lazily
only when the corresponding CLI flag is set, so ``--stt dummy`` works even if
``nemo_toolkit`` is not installed.
"""
from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from pipecat_voice.interfaces import LLMResponse, LLMToolCall, STTResult, TTSChunk


# =============================================================================
# STT
# =============================================================================


@dataclass
class DummySTT:
    """Echoes primed text. Call :meth:`prime` from the runner to inject input."""

    sample_rate_in: int = 16000
    _primed_text: str = ""
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def prime(self, text: str) -> None:
        """Queue a transcript that will be returned on the next ``transcribe`` call."""
        self._primed_text = text

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> STTResult:
        # If we have a primed phrase, return it; otherwise silence.
        async with self._lock:
            text = self._primed_text
            self._primed_text = ""
        if not text:
            return STTResult(text="", confidence=0.0, is_final=True)
        return STTResult(text=text, confidence=1.0, is_final=True)


# =============================================================================
# TTS
# =============================================================================


@dataclass
class DummyTTS:
    """Emits silence at the configured sample rate.

    Yields one ``TTSChunk`` per call, all-zero PCM. Total size is ~1 second so
    callers see the stream shape they would see from a real TTS.
    """

    sample_rate_out: int = 16000
    voice: str = "silence"
    chunk_ms: int = 100

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:
        bytes_per_chunk = self.sample_rate_out * 2 * self.chunk_ms // 1000
        # Yield at least one chunk so the pipeline sees an output.
        total_ms = max(self.chunk_ms, 200)
        n_chunks = total_ms // self.chunk_ms
        for _ in range(n_chunks):
            yield TTSChunk(pcm=b"\x00" * bytes_per_chunk, sample_rate=self.sample_rate_out, is_final=False)
        yield TTSChunk(pcm=b"\x00" * bytes_per_chunk, sample_rate=self.sample_rate_out, is_final=True)


# =============================================================================
# LLM
# =============================================================================


@dataclass
class DummyLLM:
    """Walks a scripted list of responses, then falls back to a deterministic reply.

    For tests that want to drive a specific transcript or tool-call sequence
    without contacting a real LLM.
    """

    model: str = "dummy"
    scripted: list[LLMResponse] = field(default_factory=list)
    default_text: str = "Hello, this is a dummy agent reply."
    fallback_tool_name: str = "transfer_to_human_agents"
    fallback_tool_args: dict[str, Any] = field(default_factory=lambda: {"summary": "out of scope"})

    def script(self, responses: list[LLMResponse]) -> None:
        self.scripted = list(responses)

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self.scripted:
            return self.scripted.pop(0)
        # Default: emit one tool call so the loop terminates cleanly after
        # the orchestrator sees a `###STOP###`. This keeps reward=0 but
        # proves the loop closes.
        return LLMResponse(
            content=None,
            tool_calls=[
                LLMToolCall(
                    id="dummy-0",
                    name=self.fallback_tool_name,
                    arguments=self.fallback_tool_args,
                )
            ],
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            latency_ms=0.0,
        )


# =============================================================================
# LLM scripted-response helpers (used by tests)
# =============================================================================


def scripted_text(text: str) -> LLMResponse:
    return LLMResponse(content=text, usage={"prompt_tokens": 0, "completion_tokens": 0}, latency_ms=0.0)


def scripted_tool_call(name: str, args: dict[str, Any], call_id: str = "tc") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[LLMToolCall(id=call_id, name=name, arguments=args)],
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        latency_ms=0.0,
    )


def scripted_stop(text: str = "###STOP###") -> LLMResponse:
    return scripted_text(text)

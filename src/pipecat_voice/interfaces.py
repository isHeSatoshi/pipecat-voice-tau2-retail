"""Replaceable STT/TTS/LLM boundary contracts.

Every concrete implementation (Parakeet, Chatterbox, MiniMax, dummy, ...)
satisfies one of these protocols. The agent and user pipelines depend on the
protocols, not the concrete classes, so swapping implementations is a matter
of constructing different concrete objects at runner setup time.

This module is intentionally small and free of heavy imports so it can be
used in tests and the CLI without pulling in Pipecat or audio backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable


# =============================================================================
# STT
# =============================================================================


@dataclass
class STTResult:
    """Result of transcribing one chunk of audio.

    The runner accumulates ``text`` chunks into the LLM context; ``confidence``
    is recorded to the trace if the backend exposes it.
    """

    text: str
    confidence: float = 1.0
    is_final: bool = True
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class STTProtocol(Protocol):
    """Speech-to-text service interface.

    Implementations may be local (Parakeet via nemo_toolkit) or remote
    (Whisper via OpenAI). The runner does not care.

    ``transcribe`` is called per audio chunk; for streaming backends it should
    yield ``STTResult`` instances as utterances become available.
    """

    sample_rate_in: int

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> STTResult:
        """Transcribe a chunk of PCM audio into text."""
        ...


# =============================================================================
# TTS
# =============================================================================


@dataclass
class TTSChunk:
    """One chunk of synthesised audio from the TTS backend."""

    pcm: bytes
    sample_rate: int
    is_final: bool = True


@runtime_checkable
class TTSProtocol(Protocol):
    """Text-to-speech service interface.

    ``synthesize`` yields one or more PCM chunks. Pipecat's TTSService base
    wraps this protocol into a real Pipecat ``TTSService`` instance.
    """

    sample_rate_out: int
    voice: str

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:
        """Yield PCM chunks for the given text."""
        ...


# =============================================================================
# LLM
# =============================================================================


@dataclass
class LLMToolCall:
    """One tool call emitted by an LLM.

    Mirrors Pipecat's ``FunctionCall`` shape so the runner can hand off to
    tau2's ``Environment.make_tool_call`` without translation.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM call.

    Exactly one of ``content`` (text reply) or ``tool_calls`` (one or more
    tool invocations) is non-None, mirroring tau2's message model
    (``content`` OR ``tool_calls``, never both).
    """

    content: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@runtime_checkable
class LLMProtocol(Protocol):
    """LLM service interface.

    The agent pipeline's Tau2ToolBridge turns each ``LLMResponse.tool_call``
    into a call into tau2's ``Environment``. The user pipeline ignores tool
    calls (the user simulator doesn't have tools).
    """

    model: str

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Call the LLM with the given prompt and return one response."""
        ...


# =============================================================================
# Tool bridge
# =============================================================================


@runtime_checkable
class ToolExecutor(Protocol):
    """Execute a single tool call against tau2's environment.

    The Tau2ToolBridge in the agent pipeline is a Pipecat ``FrameProcessor``
    that watches for ``FunctionCallsStartedFrame`` and dispatches each call
    through this protocol. The default implementation routes into tau2's
    ``Environment.make_tool_call``.
    """

    async def execute(self, call: LLMToolCall, requestor: str) -> str:
        """Return the tool result as a JSON string for the LLM context."""
        ...


# =============================================================================
# Sink for trace events
# =============================================================================


@runtime_checkable
class TraceSink(Protocol):
    """Receives structured events from the runner for offline analysis."""

    def emit(self, event: dict[str, Any]) -> None: ...


# Type alias for the factory callable the runner uses to instantiate services.
ServiceFactory = Callable[..., Any]

"""MiniMax (M2.7) LLM service via Pipecat's AnthropicLLMService.

MiniMax exposes an Anthropic-compatible API at
``https://api.minimax.io/anthropic``. We instantiate Pipecat's
``AnthropicLLMService`` with a custom ``AsyncAnthropic`` client whose
``base_url`` points at MiniMax and whose ``auth_token`` is the MiniMax API
key. This is the cleanest way to route through Pipecat's Anthropic adapter
without forking the service.

Environment variables
---------------------

- ``MINIMAX_API_KEY`` — the API key.
- ``MINIMAX_API_BASE`` — defaults to ``https://api.minimax.io/anthropic``.
- ``MINIMAX_AGENT_MODEL`` — defaults to ``MiniMax-M2.7``.

Lazy import of ``anthropic`` so the harness still loads without it
(useful for tests that pass ``--llm dummy``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from pipecat_voice.config import VoiceConfig


@dataclass
class MiniMaxAnthropicLLMServiceFactory:
    """Builds a Pipecat ``AnthropicLLMService`` wired to MiniMax.

    Pipecat's Anthropic service accepts a pre-constructed ``AsyncAnthropic``
    client; we use that to swap the base URL and auth token without touching
    the service's internals.
    """

    model: str
    api_key: str
    api_base: str = "https://api.minimax.io/anthropic"
    # Voice turns must be short: a 4096-token ceiling lets the LLM ramble,
    # and every extra sentence costs ~10-30 s of Chatterbox synthesis plus
    # LLM latency. 512 tokens still fits tool calls + a few sentences.
    max_tokens: int = 512
    temperature: float = 0.0
    _service: Any = field(default=None, init=False, repr=False)  # noqa: F821

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "MINIMAX_API_KEY is empty. Set it in .env before running with "
                "--agent-llm minimax. For local testing, use --agent-llm dummy."
            )

    def build(self):
        """Construct the AnthropicLLMService instance."""
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "MiniMax LLM requires the `anthropic` SDK. Install with "
                "`uv pip install anthropic`."
            ) from e
        try:
            from pipecat.services.anthropic.llm import (
                AnthropicLLMService,  # type: ignore
                AnthropicLLMSettings,  # type: ignore
            )
        except ImportError as e:
            raise RuntimeError("Pipecat's AnthropicLLMService is unavailable.") from e

        client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        logger.info(f"MiniMax Anthropic client base_url={self.api_base}")
        settings = AnthropicLLMSettings(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self._service = AnthropicLLMService(
            api_key=self.api_key,
            settings=settings,
            client=client,
        )
        return self._service


def build_minimax_llm_service(cfg: VoiceConfig, *, side: str = "agent"):
    """Convenience factory: build the MiniMax/Anthropic LLM service from a VoiceConfig."""
    if side == "agent":
        model = cfg.agent_model
    else:
        model = cfg.user_model
    factory = MiniMaxAnthropicLLMServiceFactory(
        model=model,
        api_key=cfg.minimax_api_key,
        api_base=cfg.minimax_api_base,
    )
    return factory.build()


def build_minimax_llm(cfg: VoiceConfig, *, side: str = "agent"):
    """Top-level helper used by the pipelines. Returns the LLM service.

    Falls back to the dummy LLM if ``cfg.agent_llm_impl == "dummy"`` for the
    agent side (or ``cfg.user_llm_impl`` for the user side).
    """
    impl = cfg.agent_llm_impl if side == "agent" else cfg.user_llm_impl
    if impl == "minimax":
        return build_minimax_llm_service(cfg, side=side)
    elif impl == "anthropic":
        # Use the real Anthropic SDK directly (not via MiniMax proxy).
        import os

        from anthropic import AsyncAnthropic
        from pipecat.services.anthropic.llm import (
            AnthropicLLMService,
            AnthropicLLMSettings,
        )

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = cfg.agent_model if side == "agent" else cfg.user_model
        settings = AnthropicLLMSettings(model=model, max_tokens=512, temperature=0.0)
        client = AsyncAnthropic(api_key=api_key)
        return AnthropicLLMService(api_key=api_key, settings=settings, client=client)
    elif impl == "dummy":
        # Lazy import so the dummy branch doesn't pull Pipecat services.
        from pipecat_voice.services.dummy_stt_tts_llm import DummyLLM

        return DummyLLM(model=cfg.agent_model if side == "agent" else cfg.user_model)
    else:
        raise ValueError(f"Unknown agent_llm_impl: {impl!r}")

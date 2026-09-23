"""Agent system prompt variants.

Each module under this package exposes a string ``AGENT_SYSTEM_PROMPT``
that becomes the agent's system message. The CLI flag
``--prompt-variant <name>`` selects which one to use; the default
(``baseline``) reproduces the original tau2 prompt verbatim.

Variants are *prompt-only* — no pipeline or tool changes. Adding a new
variant is as simple as dropping a new file in this directory.
"""
from __future__ import annotations

from typing import Callable, Dict

from pipecat_voice.prompts import baseline, v1, v2, v3


_REGISTRY: Dict[str, object] = {
    "baseline": baseline,
    "v1": v1,
    "v2": v2,
    "v3": v3,
}


def register(name: str, module: object) -> None:
    """Register a prompt module by name. ``module`` must expose
    ``AGENT_SYSTEM_PROMPT`` as a string."""
    if not hasattr(module, "AGENT_SYSTEM_PROMPT"):
        raise ValueError(f"Prompt module {module!r} must define AGENT_SYSTEM_PROMPT")
    _REGISTRY[name] = module


def load_agent_prompt(name: str) -> str:
    """Return the ``AGENT_SYSTEM_PROMPT`` string for ``name``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown prompt variant {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return getattr(_REGISTRY[name], "AGENT_SYSTEM_PROMPT")


def available_variants() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["load_agent_prompt", "register", "available_variants"]

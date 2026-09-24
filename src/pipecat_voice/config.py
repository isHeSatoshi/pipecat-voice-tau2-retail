"""Configuration loading and dataclasses for the pipecat_voice harness.

All configuration is env-driven so the same harness can swap STT/TTS/LLM
implementations without code changes. See ``.env.example`` for the full list.

Configuration model:
- :class:`VoiceConfig` — runtime config for one evaluation run.
- Loaded by :func:`load_config` from environment + CLI overrides.

Implementation notes
--------------------

MiniMax is reached via an Anthropic-compatible endpoint:

    https://api.minimax.io/anthropic

This means we route through Pipecat's ``AnthropicLLMService`` with a custom
``AsyncAnthropic`` client whose ``base_url`` points at MiniMax. The API key
goes through the standard ``x-api-key`` header (the Anthropic SDK handles
this). No data leaves the Anthropic SDK beyond what the MiniMax endpoint
exposes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Project root (where pyproject.toml lives). Used as a default for output dirs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VoiceConfig:
    """Resolved configuration for one eval run.

    Every field is overridable via env var (see ``_from_env``) or via the CLI
    flag parser in ``cli.py``.
    """

    # --- Domain / task selection (forwarded to tau2) ---
    domain: str = "retail"
    task_split: str = "base"
    task_ids: list[str] = field(default_factory=list)
    max_tasks: Optional[int] = None

    # --- LLM routing ---
    agent_llm_impl: str = "minimax"  # minimax | anthropic | openai | dummy
    user_llm_impl: str = "minimax"

    agent_model: str = "MiniMax-M2.7"
    user_model: str = "MiniMax-M2.7"

    minimax_api_key: str = ""
    minimax_api_base: str = "https://api.minimax.io/anthropic"

    # --- STT ---
    stt_impl: str = "parakeet"  # parakeet | whisper | dummy
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3"

    # --- TTS ---
    tts_impl: str = "chatterbox"  # chatterbox | elevenlabs | dummy
    chatterbox_voice: str = "default"

    # --- Audio pipeline ---
    sample_rate: int = 16000
    max_conversation_seconds: int = 240

    # --- Run bookkeeping ---
    seed: int = 42
    out_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "runs")
    run_name: str = "default"

    # --- Misc ---
    enable_vad: bool = True
    enable_metrics: bool = True

    # --- Prompt overrides (set by CLI --prompt-variant) ---
    agent_system_prompt_override: Optional[str] = None
    prompt_variant: str = "baseline"

    def __post_init__(self) -> None:
        # Normalise paths.
        self.out_dir = Path(self.out_dir)
        if not self.task_ids:
            self.task_ids = []

    @property
    def audio_bus_sample_rate(self) -> int:
        """Sample rate used on the in-memory audio bus between pipelines."""
        return self.sample_rate

    @property
    def tau2_user_instructions(self) -> Optional[str]:
        """If set, replaces the user_simulator system prompt's instructions block.

        Used in tests; in production, the runner reads ``task.user_scenario``.
        """
        return None

    @classmethod
    def _from_env(cls) -> "VoiceConfig":
        # Load env in two passes so we pick up keys from both:
        #   - pipecat_voice/.env   (this project's own vars)
        #   - ../tau2-bench/.env   (the user's tau2 key, e.g. ANTHROPIC_API_KEY)
        # Python-dotenv defaults to NOT overriding real env vars, so a key
        # already set in the shell wins over either file.
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        sibling_tau2_env = PROJECT_ROOT.parent / "tau2-bench" / ".env"
        if sibling_tau2_env.exists():
            load_dotenv(sibling_tau2_env, override=False)

        def _env(name: str, default: str) -> str:
            return os.environ.get(name, default)

        def _env_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        def _env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        # MiniMax is reached via the Anthropic-compatible endpoint. The key
        # the user pasted into tau2-bench/.env is ANTHROPIC_API_KEY; we
        # accept MINIMAX_API_KEY as an override if both are present.
        minimax_api_key = _env("MINIMAX_API_KEY", "") or _env("ANTHROPIC_API_KEY", "")
        minimax_api_base = (
            _env("MINIMAX_API_BASE", "")
            or _env("ANTHROPIC_API_BASE", "")
            or "https://api.minimax.io/anthropic"
        )

        cfg = cls(
            domain=_env("TAU2_DOMAIN", "retail"),
            task_split=_env("TAU2_TASK_SPLIT", "base"),
            agent_llm_impl=_env("PIPECAT_AGENT_LLM_IMPL", "minimax"),
            user_llm_impl=_env("PIPECAT_USER_LLM_IMPL", "minimax"),
            agent_model=_env("MINIMAX_AGENT_MODEL", "MiniMax-M2.7"),
            user_model=_env("MINIMAX_USER_MODEL", "MiniMax-M2.7"),
            minimax_api_key=minimax_api_key,
            minimax_api_base=minimax_api_base,
            stt_impl=_env("PIPECAT_VOICE_STT", "parakeet"),
            tts_impl=_env("PIPECAT_VOICE_TTS", "chatterbox"),
            parakeet_model=_env(
                "PIPECAT_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3"
            ),
            chatterbox_voice=_env("PIPECAT_CHATTERBOX_VOICE", "default"),
            sample_rate=_env_int("PIPECAT_VOICE_SAMPLE_RATE", 16000),
            max_conversation_seconds=_env_int("PIPECAT_VOICE_MAX_SECONDS", 240),
            seed=_env_int("PIPECAT_VOICE_SEED", 42),
            out_dir=Path(
                _env("PIPECAT_VOICE_OUT_DIR", str(PROJECT_ROOT / "data" / "runs"))
            ),
            run_name=_env("PIPECAT_VOICE_RUN_NAME", "default"),
            enable_vad=_env_bool("PIPECAT_VOICE_VAD", True),
            enable_metrics=_env_bool("PIPECAT_VOICE_METRICS", True),
            prompt_variant=_env("PIPECAT_AGENT_PROMPT", "baseline"),
        )
        if (
            cfg.prompt_variant != "baseline"
            and cfg.agent_system_prompt_override is None
        ):
            from pipecat_voice.prompts import load_agent_prompt

            cfg.agent_system_prompt_override = load_agent_prompt(cfg.prompt_variant)
        return cfg

    def merge_cli(self, **overrides) -> "VoiceConfig":
        """Return a copy with CLI overrides applied (only known fields)."""
        valid = {f for f in self.__dataclass_fields__}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        if not filtered:
            return self
        from dataclasses import replace

        return replace(self, **filtered)


def load_config() -> VoiceConfig:
    """Load VoiceConfig from environment (call once at startup)."""
    return VoiceConfig._from_env()

"""Tau2EvalRunner — drives the closed-loop Pipecat eval harness end to end.

Responsibilities
----------------

1. Build a tau2 ``Environment`` for the task (DB + policy + tools).
2. Build the agent and user Pipecat pipelines.
3. Wire them together via the in-memory ``AudioBus``.
4. Run both pipelines concurrently until:
   - The user simulator emits ``###STOP###`` (or related signals)
   - The conversation runs past ``max_conversation_seconds``
   - Either pipeline raises an unrecoverable error
5. Translate the captured LLM context into a tau2 ``SimulationRun``.
6. Call tau2's ``evaluate_simulation`` and attach ``reward_info``.
7. Write the per-task trace + the canonical ``trajectory.json``.

Why this lives in the ``tau2`` submodule
----------------------------------------

The runner is the only piece that knows about tau2's ``Environment``,
``Task``, ``SimulationRun``, and ``evaluate_simulation``. Everything else
above it (transport, pipelines, services, observability) is tau2-agnostic.

Pipecat 1.x note
----------------

This module uses Pipecat's ``PipelineWorker`` (which ``PipelineTask``
now aliases). Each worker is started with ``runner.run(worker)``; the
runner manages the event loop and signal handling.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.services.llm_service import LLMService

from pipecat_voice.config import VoiceConfig
from pipecat_voice.observability.frame_observer import VoiceTraceObserver
from pipecat_voice.observability.tau2_bridge import (
    build_simulation_run,
    run_evaluator_local,
)
from pipecat_voice.observability.trace_writer import TraceWriter
from pipecat_voice.pipelines.agent_pipeline import build_agent_pipeline
from pipecat_voice.pipelines.pipecat_adapters import (
    ProtocolSTTService,
    ProtocolTTSService,
)
from pipecat_voice.pipelines.user_pipeline import build_user_pipeline
from pipecat_voice.services import (
    ChatterboxTTS,
    DummyLLM,
    DummySTT,
    DummyTTS,
    ParakeetSTT,
    build_minimax_llm,
)
from pipecat_voice.tau2.environment import build_tau2_environment
from pipecat_voice.transport.virtual_transport import (
    AudioBus,
    VirtualTransport,
    VirtualTransportParams,
    make_default_vad,
)

# =============================================================================
# Service factories
# =============================================================================


def _build_stt(cfg: VoiceConfig, *, side: str):
    if cfg.stt_impl == "parakeet":
        return ParakeetSTT(
            sample_rate_in=cfg.sample_rate, model_name=cfg.parakeet_model
        )
    elif cfg.stt_impl == "whisper":
        # Whisper is wired through Pipecat's WhisperSTTService directly; we
        # import lazily here so the harness stays importable without openai.
        from pipecat_voice.services.whisper_stt import WhisperSTT  # type: ignore

        return WhisperSTT(sample_rate_in=cfg.sample_rate)
    elif cfg.stt_impl == "dummy":
        return DummySTT(sample_rate_in=cfg.sample_rate)
    raise ValueError(f"Unknown STT impl: {cfg.stt_impl!r}")


def _build_tts(cfg: VoiceConfig, *, side: str):
    if cfg.tts_impl == "chatterbox":
        return ChatterboxTTS(
            sample_rate_out=cfg.sample_rate, voice=cfg.chatterbox_voice
        )
    elif cfg.tts_impl == "elevenlabs":
        from pipecat_voice.services.elevenlabs_tts import ElevenLabsTTS  # type: ignore

        return ElevenLabsTTS(sample_rate_out=cfg.sample_rate)
    elif cfg.tts_impl == "dummy":
        return DummyTTS(sample_rate_out=cfg.sample_rate)
    raise ValueError(f"Unknown TTS impl: {cfg.tts_impl!r}")


def _cached_stt_impl(self: Tau2EvalRunner, cfg: VoiceConfig, *, side: str):
    key = (cfg.stt_impl, cfg.sample_rate, cfg.parakeet_model)
    if key not in self._stt_impl_cache:
        self._stt_impl_cache[key] = _build_stt(cfg, side=side)
    return self._stt_impl_cache[key]


def _cached_tts_impl(self: Tau2EvalRunner, cfg: VoiceConfig, *, side: str):
    key = (cfg.tts_impl, cfg.sample_rate, cfg.chatterbox_voice)
    if key not in self._tts_impl_cache:
        self._tts_impl_cache[key] = _build_tts(cfg, side=side)
    return self._tts_impl_cache[key]


def _build_llm(cfg: VoiceConfig, *, side: str):
    if (side == "agent" and cfg.agent_llm_impl == "dummy") or (
        side == "user" and cfg.user_llm_impl == "dummy"
    ):
        return DummyLLM(model=cfg.agent_model if side == "agent" else cfg.user_model)
    return build_minimax_llm(cfg, side=side)


# =============================================================================
# Runner
# =============================================================================


@dataclass
class Tau2EvalRunner:
    """Drives one tau2 task through the Pipecat closed-loop harness.

    Construction is side-effect-free. Call :meth:`run` to execute one task
    end to end (build pipelines → run → score → persist traces).
    """

    cfg: VoiceConfig
    trace_dir: Path = field(default_factory=lambda: Path("data/runs"))
    enable_vad: bool = True
    _stt_impl_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _tts_impl_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def run(self, task) -> dict[str, Any]:
        """Execute one task. Returns a dict with reward + trace paths."""
        # Wire the optional prompt-variant override into the context builder.
        from pipecat_voice.tau2.context import set_agent_system_prompt_override

        override = getattr(self.cfg, "agent_system_prompt_override", None)
        set_agent_system_prompt_override(override)
        try:
            return asyncio.run(self._run_async(task))
        finally:
            set_agent_system_prompt_override(None)

    async def _run_async(self, task) -> dict[str, Any]:
        run_id = f"sim_{uuid.uuid4().hex[:8]}"
        sim_dir = Path(self.trace_dir) / f"task_{task.id}" / run_id
        sim_dir.mkdir(parents=True, exist_ok=True)
        trace = TraceWriter(out_dir=sim_dir, task_id=task.id)
        prompt_text = self.cfg.agent_system_prompt_override or "baseline"
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        trace.emit_meta(
            {
                "domain": self.cfg.domain,
                "split": self.cfg.task_split,
                "agent_llm_impl": self.cfg.agent_llm_impl,
                "user_llm_impl": self.cfg.user_llm_impl,
                "agent_model": self.cfg.agent_model,
                "user_model": self.cfg.user_model,
                "stt_impl": self.cfg.stt_impl,
                "tts_impl": self.cfg.tts_impl,
                "sample_rate": self.cfg.sample_rate,
                "max_conversation_seconds": self.cfg.max_conversation_seconds,
                "seed": self.cfg.seed,
                "minimax_api_base": self.cfg.minimax_api_base,
                "run_id": run_id,
                "prompt": self.cfg.prompt_variant,
                "prompt_sha256": prompt_hash,
                "vad_enabled": self.enable_vad,
                "tool_execution_policy": "single_serialized_exact_signature",
            }
        )

        t0 = time.time()

        # 1. Build tau2 environment for this task.
        env = build_tau2_environment(self.cfg.domain)
        # Replay the task's initial_state so DB + history match spec.
        if task.initial_state is not None:
            env.set_state(
                initialization_data=task.initial_state.initialization_data,
                initialization_actions=task.initial_state.initialization_actions,
                message_history=task.initial_state.message_history or [],
                strict=False,  # avoid aborting on minor cosmetic drift in recorded outputs
            )

        # 2. Build the audio bus + transports.
        bus = AudioBus(sample_rate=self.cfg.sample_rate)
        vad_agent = (
            make_default_vad(sample_rate=self.cfg.sample_rate)
            if self.enable_vad
            else None
        )
        vad_user = (
            make_default_vad(sample_rate=self.cfg.sample_rate)
            if self.enable_vad
            else None
        )
        agent_transport = VirtualTransport(
            params=VirtualTransportParams(
                bus=bus,
                direction="b_to_a",  # agent reads from b_to_a (user's output)
                vad_analyzer=vad_agent,
                sample_rate=self.cfg.sample_rate,
                audio_path=sim_dir / "agent_audio.wav",
            )
        )
        user_transport = VirtualTransport(
            params=VirtualTransportParams(
                bus=bus,
                direction="a_to_b",  # user reads from a_to_b (agent's output)
                vad_analyzer=vad_user,
                sample_rate=self.cfg.sample_rate,
                audio_path=sim_dir / "user_audio.wav",
            )
        )

        # 3. Build STT / TTS / LLM services.
        agent_stt = ProtocolSTTService(
            stt_impl=_cached_stt_impl(self, self.cfg, side="agent"),
            sample_rate=self.cfg.sample_rate,
            name="agent-stt",
        )
        agent_tts = ProtocolTTSService(
            tts_impl=_cached_tts_impl(self, self.cfg, side="agent"),
            sample_rate=self.cfg.sample_rate,
            name="agent-tts",
        )
        agent_llm = _build_llm(self.cfg, side="agent")
        # Wrap in a Pipecat LLMService if it's not already one.
        if not isinstance(agent_llm, LLMService):
            from pipecat_voice.pipelines.pipecat_adapters import ProtocolLLMService

            agent_llm = ProtocolLLMService(llm_impl=agent_llm, name="agent-llm")

        user_stt = ProtocolSTTService(
            stt_impl=_cached_stt_impl(self, self.cfg, side="user"),
            sample_rate=self.cfg.sample_rate,
            name="user-stt",
        )
        user_tts = ProtocolTTSService(
            tts_impl=_cached_tts_impl(self, self.cfg, side="user"),
            sample_rate=self.cfg.sample_rate,
            name="user-tts",
        )
        user_llm = _build_llm(self.cfg, side="user")
        if not isinstance(user_llm, LLMService):
            from pipecat_voice.pipelines.pipecat_adapters import ProtocolLLMService

            user_llm = ProtocolLLMService(llm_impl=user_llm, name="user-llm")

        # 3a. Warm up heavy local models (Parakeet, Chatterbox) so the
        # first conversation turn doesn't pay first-time GPU/CUDA load
        # latency (and so load failures surface before the pipelines start).
        # Dummy impls have no `_ensure_model` and fall back to a no-op.
        # Sequential (not gather): NeMo/Chatterbox imports + CUDA init are
        # not thread-safe on first load; concurrent warm-up caused spurious
        # ImportErrors. Second instance of each impl hits the cache and is fast.
        warmup_errors: list[str] = []
        for label, svc in (("agent-stt", agent_stt), ("user-stt", user_stt)):
            try:
                await _warmup_stt(svc, self.cfg.sample_rate)
            except Exception as e:
                warmup_errors.append(f"{label} warm-up: {e}")
                logger.warning(f"{label} warm-up failed: {e}")
        for label, svc in (("agent-tts", agent_tts), ("user-tts", user_tts)):
            try:
                await _warmup_tts(svc, self.cfg.sample_rate)
            except Exception as e:
                warmup_errors.append(f"{label} warm-up: {e}")
                logger.warning(f"{label} warm-up failed: {e}")
        logger.info("STT/TTS warm-up complete")
        # Conversation clock starts after warm-up so model-load time isn't
        # billed against the conversation budget.
        t0 = time.time()

        # 4. Build pipelines.
        stop_event = asyncio.Event()
        agent_parts = build_agent_pipeline(
            cfg=self.cfg,
            transport=agent_transport,
            stt_service=agent_stt,
            llm_service=agent_llm,
            tts_service=agent_tts,
            env=env,
            task=task,
            trace=trace,
        )
        user_parts = build_user_pipeline(
            cfg=self.cfg,
            transport=user_transport,
            stt_service=user_stt,
            llm_service=user_llm,
            tts_service=user_tts,
            task=task,
            stop_event=stop_event,
        )

        # 5. Build PipelineWorkers and run them in one runner.
        # ``start_timeout_secs`` lives on the worker, not on PipelineParams.
        # 60 s is generous: heavy models are warmed up above and SmartTurn's
        # ONNX load happens at aggregator construction. Anything slower is a
        # real stall and should fail fast, not hang for 5 minutes.
        # ``idle_timeout_secs`` is disabled: eval turns (LLM + tool calls +
        # TTS) can be quiet for minutes without the pipeline being dead; the
        # runner's own max_conversation_seconds bounds the run instead.
        agent_worker = PipelineWorker(
            agent_parts.pipeline,
            params=PipelineParams(allow_interruptions=True),
            start_timeout_secs=60.0,
            setup_timeout_secs=60.0,
            idle_timeout_secs=None,
            name=f"agent-worker-{task.id}",
        )
        user_worker = PipelineWorker(
            user_parts.pipeline,
            params=PipelineParams(allow_interruptions=True),
            start_timeout_secs=60.0,
            setup_timeout_secs=60.0,
            idle_timeout_secs=None,
            name=f"user-worker-{task.id}",
        )

        runner = PipelineRunner()
        agent_observer = VoiceTraceObserver(trace, "agent")
        user_observer = VoiceTraceObserver(trace, "user")
        agent_worker.add_observer(agent_observer)
        user_worker.add_observer(user_observer)
        # Register both workers with a single runner. Pipecat's runner is
        # *not* safe to share across multiple ``runner.run(...)`` calls
        # because its ``_entries`` dict is mutated concurrently by the
        # workers themselves; running all of them through one runner
        # avoids the race.
        await runner.add_workers(agent_worker, user_worker)
        trace.reset_clock()
        trace.emit({"type": "run_start", "task_id": task.id})

        async def _kick_user():
            await asyncio.sleep(0.5)
            try:
                await user_worker.queue_frame(_llm_run_frame())
            except Exception as e:
                logger.warning(f"Failed to kick user pipeline: {e}")

        stop_kind: dict[str, str] = {"reason": "agent_stop"}
        deadline = time.monotonic() + self.cfg.max_conversation_seconds
        last_message_count = len(agent_parts.context.messages)
        last_context_change = time.monotonic()

        async def _stopper():
            nonlocal last_message_count, last_context_change
            while True:
                if stop_event.is_set():
                    stop_kind["reason"] = "user_stop"
                    break
                now = time.monotonic()
                message_count = len(agent_parts.context.messages)
                if message_count != last_message_count:
                    last_message_count = message_count
                    last_context_change = now
                if (
                    agent_parts.tool_policy is not None
                    and agent_parts.tool_policy.has_successful_write
                    and now - last_context_change >= 12.0
                ):
                    stop_kind["reason"] = "agent_stop"
                    stop_kind["basis"] = "post_write_inactivity"
                    break
                if now >= deadline:
                    stop_kind["reason"] = "timeout"
                    logger.warning(
                        f"Conversation timed out after {self.cfg.max_conversation_seconds}s"
                    )
                    break
                await asyncio.sleep(0.5)
            trace.emit({"type": "stop_signal", **stop_kind})
            try:
                await runner.cancel()
            except Exception:
                pass

        # Only the user side is kicked: its greeting bootstraps the loop
        # (agent greeting is already first in the agent context; kicking the
        # agent too makes it reply to an empty context and wastes a full TTS
        # turn on "your message is empty").
        run_error: str | None = None
        try:
            await asyncio.gather(
                runner.run(),
                _stopper(),
                _kick_user(),
                return_exceptions=False,
            )
        except Exception as e:
            run_error = str(e)
            logger.exception(f"Pipeline run failed: {e}")
            trace.emit({"type": "error", "error": run_error})

        component_errors = [
            error
            for error in (
                run_error,
                *warmup_errors,
                getattr(agent_stt, "last_error", None),
                getattr(user_stt, "last_error", None),
                getattr(agent_tts, "last_error", None),
                getattr(user_tts, "last_error", None),
                getattr(agent_transport.input(), "last_error", None),
                getattr(user_transport.input(), "last_error", None),
                *agent_observer.errors,
                *user_observer.errors,
            )
            if error
        ]
        if component_errors:
            run_error = "; ".join(dict.fromkeys(component_errors))

        duration = time.time() - t0
        termination_reason = _detect_termination(
            context_messages=_context_snapshot(user_parts.context),
            forced_reason=stop_kind["reason"],
        )
        # Score the AGENT context: it holds the assistant turns + tau2 tool
        # calls the evaluator checks. (The user context mirrors the same
        # conversation from the simulator side and has no tool calls.)
        simulation_run = build_simulation_run(
            task=task,
            context_messages=_context_snapshot(agent_parts.context),
            termination_reason=termination_reason,
            duration_seconds=duration,
            seed=self.cfg.seed,
            extra_info={
                "prompt_sha256": prompt_hash,
                "vad_enabled": self.enable_vad,
                "infrastructure_error": run_error,
            },
        )
        # Close the bus so the pipelines shut down cleanly if any background
        # task is still draining.
        bus.closed = True
        # Score with tau2's local evaluators (ENV + ACTION + COMMUNICATE;
        # the NL judge needs an OpenAI key we don't have — see
        # run_evaluator_local).
        try:
            reward_info = run_evaluator_local(simulation_run, task)
        except Exception as e:
            logger.warning(f"Evaluator failed: {e}")
            reward_info = None
            run_error = "; ".join(filter(None, [run_error, f"evaluator: {e}"]))

        # Attach reward_info to the run.
        if reward_info is not None:
            try:
                simulation_run.reward_info = reward_info
            except Exception:
                pass
        simulation_run.info["infrastructure_error"] = run_error

        try:
            from pipecat_voice.eval import run_all_checks

            check_outcome = run_all_checks(simulation_run, task)
            check_results = [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "message": result.message,
                }
                for result in check_outcome.results
            ]
        except Exception as e:
            check_results = [
                {"name": "check_error", "passed": False, "message": str(e)}
            ]

        trace.dump_simulation_run(simulation_run)
        trace.emit(
            {
                "type": "run_end",
                "duration_seconds": duration,
                "termination": termination_reason.value,
                "checks": check_results,
            }
        )
        trace.close()

        # Dummy services intentionally emit no persisted WAVs. Keep the
        # offline smoke path successful while still surfacing missing audio
        # for the real voice stack.
        if self.cfg.stt_impl == "dummy" and self.cfg.tts_impl == "dummy":
            logger.info("Skipping audio artifacts for dummy STT/TTS services")
        else:
            try:
                from pipecat_voice.observability.audio_artifacts import build_artifacts

                build_artifacts(sim_dir)
            except Exception as e:
                run_error = "; ".join(filter(None, [run_error, f"audio artifacts: {e}"]))

        reward_info_dict = (
            reward_info.model_dump(mode="json") if reward_info is not None else {}
        )
        reward_details = reward_info_dict.get("info") or {}
        action_checks = reward_info_dict.get("action_checks") or []
        db_check = reward_info_dict.get("db_check") or {}
        return {
            "task_id": task.id,
            "reward": getattr(reward_info, "reward", None),
            "local_reward": reward_details.get("partial_reward"),
            "strict_reward_available": reward_details.get("strict_reward_available"),
            "db_match": db_check.get("db_match"),
            "actions_matched": sum(
                1 for action in action_checks if action.get("action_match")
            ),
            "actions_total": len(action_checks),
            "termination_reason": termination_reason.value,
            "duration_seconds": duration,
            "trajectory_path": str(trace.trajectory_path.resolve()),
            "trace_path": str(trace.trace_path.resolve()),
            "agent_audio_path": str((sim_dir / "agent_audio.wav").resolve()),
            "user_audio_path": str((sim_dir / "user_audio.wav").resolve()),
            "conversation_audio_path": str((sim_dir / "conversation.wav").resolve()),
            "audio_segments_path": str((sim_dir / "audio_segments.json").resolve()),
            "checks": check_results,
            "infrastructure_error": run_error,
        }


# =============================================================================
# Helpers
# =============================================================================


def _llm_run_frame():
    from pipecat.frames.frames import LLMRunFrame

    return LLMRunFrame()


async def _warmup_stt(stt_service, sample_rate: int) -> None:
    """Force-load the STT model before the pipelines start.

    Prefers the impl's ``_ensure_model`` (loads weights without paying for
    a throwaway inference); falls back to transcribing 0.5 s of silence
    for impls without an explicit ensure hook.
    """
    try:
        impl = getattr(stt_service, "_impl", None)
        ensure = getattr(impl, "_ensure_model", None)
        if callable(ensure):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, ensure)
            logger.debug("STT model warmed up")
            return
        silence = b"\x00" * (sample_rate * 2 // 2)  # 500 ms of silence
        async for _ in stt_service.run_stt(silence):
            pass
    except Exception as e:
        logger.warning(f"STT warm-up failed: {e}")
        raise


async def _warmup_tts(tts_service, sample_rate: int) -> None:
    """Force-load the TTS model weights without paying for inference.

    Uses the impl's ``_ensure_model`` when available (Chatterbox
    ``from_pretrained``); falls back to synthesising one short phrase.
    """
    try:
        impl = getattr(tts_service, "_impl", None)
        # Prefer the async loader (pins GPU work to the right thread);
        # fall back to the sync hook, then to a throwaway synthesis.
        ensure_async = getattr(impl, "ensure_loaded", None)
        if callable(ensure_async):
            await ensure_async()
            logger.debug("TTS model warmed up")
            return
        ensure = getattr(impl, "_ensure_model", None)
        if callable(ensure):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, ensure)
            logger.debug("TTS model warmed up")
            return
        async for _ in tts_service.run_tts("ok"):
            pass
    except Exception as e:
        logger.warning(f"TTS warm-up failed: {e}")
        raise


def _context_snapshot(context) -> list[dict[str, Any]]:
    """Return a copy of the LLMContext messages as plain dicts.

    Safe against concurrent mutation: retries on ``RuntimeError`` (the
    Python dict-mutated-during-iteration signal) and returns whatever
    snapshot succeeded.
    """
    import time as _time

    for _ in range(5):
        try:
            out: list[dict[str, Any]] = []
            for m in context.messages:
                if isinstance(m, dict):
                    out.append(dict(m))
                else:
                    # LLMSpecificMessage or other provider blobs: keep a tagged
                    # repr so the snapshot never crashes; the tau2 bridge
                    # drops unknown roles.
                    out.append(
                        {
                            "role": "unknown",
                            "content": repr(m),
                            "_raw_type": type(m).__name__,
                        }
                    )
            return out
        except RuntimeError:
            _time.sleep(0.01)
        except Exception:
            return []
    return []


def _detect_termination(
    context_messages: list[dict[str, Any]],
    *,
    forced_reason: str = "agent_stop",
):
    """Inspect the user-side LLM context for stop signals."""
    from tau2.data_model.simulation import TerminationReason

    if forced_reason == "timeout":
        return TerminationReason.TIMEOUT
    if forced_reason == "user_stop":
        return TerminationReason.USER_STOP
    for m in reversed(context_messages[-4:]):
        content = (m.get("content") or "").strip()
        if "###STOP###" in content:
            return TerminationReason.USER_STOP
        if "###TRANSFER###" in content or "###OUT-OF-SCOPE###" in content:
            return TerminationReason.USER_STOP
    return TerminationReason.AGENT_STOP

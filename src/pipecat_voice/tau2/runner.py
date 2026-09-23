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
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.services.llm_service import LLMService

from pipecat_voice.config import VoiceConfig
from pipecat_voice.observability.tau2_bridge import (
    build_simulation_run,
    run_evaluator_local,
)
from pipecat_voice.observability.trace_writer import TraceWriter
from pipecat_voice.pipelines.agent_pipeline import build_agent_pipeline
from pipecat_voice.pipelines.pipecat_adapters import ProtocolSTTService, ProtocolTTSService
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
        return ParakeetSTT(sample_rate_in=cfg.sample_rate, model_name=cfg.parakeet_model)
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
        return ChatterboxTTS(sample_rate_out=cfg.sample_rate, voice=cfg.chatterbox_voice)
    elif cfg.tts_impl == "elevenlabs":
        from pipecat_voice.services.elevenlabs_tts import ElevenLabsTTS  # type: ignore
        return ElevenLabsTTS(sample_rate_out=cfg.sample_rate)
    elif cfg.tts_impl == "dummy":
        return DummyTTS(sample_rate_out=cfg.sample_rate)
    raise ValueError(f"Unknown TTS impl: {cfg.tts_impl!r}")


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
        trace.emit_meta({
            "domain": self.cfg.domain,
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
        })

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
        vad_agent = make_default_vad(sample_rate=self.cfg.sample_rate) if self.enable_vad else None
        vad_user = make_default_vad(sample_rate=self.cfg.sample_rate) if self.enable_vad else None
        agent_transport = VirtualTransport(
            params=VirtualTransportParams(
                bus=bus,
                direction="b_to_a",        # agent reads from b_to_a (user's output)
                vad_analyzer=vad_agent,
                sample_rate=self.cfg.sample_rate,
            )
        )
        user_transport = VirtualTransport(
            params=VirtualTransportParams(
                bus=bus,
                direction="a_to_b",        # user reads from a_to_b (agent's output)
                vad_analyzer=vad_user,
                sample_rate=self.cfg.sample_rate,
            )
        )

        # 3. Build STT / TTS / LLM services.
        agent_stt = ProtocolSTTService(stt_impl=_build_stt(self.cfg, side="agent"), sample_rate=self.cfg.sample_rate, name="agent-stt")
        agent_tts = ProtocolTTSService(tts_impl=_build_tts(self.cfg, side="agent"), sample_rate=self.cfg.sample_rate, name="agent-tts")
        agent_llm = _build_llm(self.cfg, side="agent")
        # Wrap in a Pipecat LLMService if it's not already one.
        if not isinstance(agent_llm, LLMService):
            from pipecat_voice.pipelines.pipecat_adapters import ProtocolLLMService
            agent_llm = ProtocolLLMService(llm_impl=agent_llm, name="agent-llm")

        user_stt = ProtocolSTTService(stt_impl=_build_stt(self.cfg, side="user"), sample_rate=self.cfg.sample_rate, name="user-stt")
        user_tts = ProtocolTTSService(tts_impl=_build_tts(self.cfg, side="user"), sample_rate=self.cfg.sample_rate, name="user-tts")
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
        for svc in (agent_stt, user_stt):
            try:
                await _warmup_stt(svc, self.cfg.sample_rate)
            except Exception as e:
                logger.debug(f"STT warm-up failed: {e}")
        for svc in (agent_tts, user_tts):
            try:
                await _warmup_tts(svc, self.cfg.sample_rate)
            except Exception as e:
                logger.debug(f"TTS warm-up failed: {e}")
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
        # Register both workers with a single runner. Pipecat's runner is
        # *not* safe to share across multiple ``runner.run(...)`` calls
        # because its ``_entries`` dict is mutated concurrently by the
        # workers themselves; running all of them through one runner
        # avoids the race.
        await runner.add_workers(agent_worker, user_worker)
        trace.emit({"type": "run_start", "task_id": task.id})

        async def _kick_user():
            await asyncio.sleep(0.5)
            try:
                await user_worker.queue_frame(_llm_run_frame())
            except Exception as e:
                logger.warning(f"Failed to kick user pipeline: {e}")

        async def _stopper():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.cfg.max_conversation_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Conversation timed out after {self.cfg.max_conversation_seconds}s")
            finally:
                trace.emit({"type": "stop_signal"})
                try:
                    await runner.cancel()
                except Exception:
                    pass

        # Only the user side is kicked: its greeting bootstraps the loop
        # (agent greeting is already first in the agent context; kicking the
        # agent too makes it reply to an empty context and wastes a full TTS
        # turn on "your message is empty").
        try:
            await asyncio.gather(
                runner.run(),
                _stopper(),
                _kick_user(),
                return_exceptions=False,
            )
        except Exception as e:
            logger.exception(f"Pipeline run failed: {e}")
            trace.emit({"type": "error", "error": str(e)})

        duration = time.time() - t0
        termination_reason = _detect_termination(context_messages=_context_snapshot(user_parts.context))
        # Score the AGENT context: it holds the assistant turns + tau2 tool
        # calls the evaluator checks. (The user context mirrors the same
        # conversation from the simulator side and has no tool calls.)
        simulation_run = build_simulation_run(
            task=task,
            context_messages=_context_snapshot(agent_parts.context),
            termination_reason=termination_reason,
            duration_seconds=duration,
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

        # Attach reward_info to the run.
        if reward_info is not None:
            try:
                simulation_run.reward_info = reward_info
            except Exception:
                pass

        trace.dump_simulation_run(simulation_run)
        trace.emit({"type": "run_end", "duration_seconds": duration, "termination": termination_reason.value})
        trace.close()

        return {
            "task_id": task.id,
            "reward": getattr(reward_info, "reward", None),
            "termination_reason": termination_reason.value,
            "duration_seconds": duration,
            "trajectory_path": str(trace.trajectory_path),
            "trace_path": str(trace.trace_path),
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
        logger.debug(f"STT warm-up skipped: {e}")


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
        logger.debug(f"TTS warm-up skipped: {e}")


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


def _detect_termination(context_messages: list[dict[str, Any]]):
    """Inspect the user-side LLM context for stop signals."""
    from tau2.data_model.simulation import TerminationReason

    for m in reversed(context_messages[-4:]):
        content = (m.get("content") or "").strip()
        if "###STOP###" in content:
            return TerminationReason.USER_STOP
        if "###TRANSFER###" in content or "###OUT-OF-SCOPE###" in content:
            return TerminationReason.USER_STOP
    return TerminationReason.AGENT_STOP

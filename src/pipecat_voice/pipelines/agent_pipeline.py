"""Agent Pipecat pipeline (full-duplex voice).

Topology:

    transport.input  ->  STT  ->  user_context_aggregator  ->  LLM  ->  assistant_context_aggregator  ->  TTS  ->  transport.output
                                                ^                                                                 |
                                                |--- tools registered here (tau2 env)  -----------------------|

The pipeline consumes PCM from the ``AudioBus`` (user's TTS output) via the
``VirtualTransport.input()``, runs STT, feeds transcripts into the
``LLMContext`` via ``LLMUserAggregator``, and lets the LLM service stream a
reply. Tool calls are dispatched through Pipecat's ``FunctionCallInProgress``
flow — each tau2 tool's handler invokes ``Environment.make_tool_call``.

Why this layout?

Pipecat 1.x's standard ``Pipeline(task)`` with
``LLMUserAggregator`` / ``LLMAssistantAggregator`` gives us:

- Natural interruption: when ``UserStartedSpeakingFrame`` arrives mid-response,
  the assistant aggregator flushes its current turn and the LLM is interrupted.
- TTS streaming: the assistant aggregator emits ``LLMTextFrame`` tokens that
  the TTS service converts into PCM frames as fast as they arrive.
- Function-call parallelism: tool execution runs in parallel with token
  generation (Pipecat's ``run_in_parallel=True`` default).

We let Pipecat drive everything; the only custom logic is the
``Tau2ToolExecutor`` registered as the ``handler`` on each tool schema and
the ``TraceObserver`` that records events to JSONL.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from pipecat.frames.frames import EndFrame, Frame, LLMRunFrame, StartFrame, StopTaskFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from pipecat_voice.config import VoiceConfig
from pipecat_voice.observability.trace_writer import TraceWriter
from pipecat_voice.transport.virtual_transport import (
    VirtualTransport,
    VirtualTransportParams,
)


def _build_tools_and_context_lazy(env, task):
    """Lazy import to avoid a circular dep with pipecat_voice.tau2.__init__."""
    from pipecat_voice.tau2.context import build_tools_and_context
    return build_tools_and_context(env, task)


@dataclass
class AgentPipelineParts:
    """Bundle of constructed pieces the runner needs to start the pipeline."""

    pipeline: BasePipeline
    context: LLMContext
    transport: VirtualTransport


def _wrap_stt(stt_service, name: str) -> STTService:
    """Tag the STT service with a stable name for the trace observer."""
    stt_service._name = name
    return stt_service


def _wrap_tts(tts_service, name: str) -> TTSService:
    tts_service._name = name
    return tts_service


def build_agent_pipeline(
    *,
    cfg: VoiceConfig,
    transport: VirtualTransport,
    stt_service: STTService,
    llm_service: Any,
    tts_service: TTSService,
    env,  # tau2 Environment (kept around so the LLMContext schema handlers can dispatch)
    task,  # tau2 Task
    trace: TraceWriter,
) -> AgentPipelineParts:
    """Construct the agent's Pipecat pipeline.

    Args:
        cfg: Loaded :class:`VoiceConfig`.
        transport: The agent's :class:`VirtualTransport` (reads user audio,
            writes agent audio).
        stt_service: A Pipecat ``STTService`` (Parakeet / dummy / etc.).
        llm_service: A Pipecat LLM service (Anthropic / dummy / etc.).
        tts_service: A Pipecat ``TTSService`` (Chatterbox / dummy / etc.).
        env: The tau2 ``Environment`` for this task.
        task: The tau2 ``Task`` being evaluated.
        trace: A :class:`TraceWriter` that records per-frame events.

    Returns:
        :class:`AgentPipelineParts` containing the constructed pipeline +
        LLM context + transport. The runner wires these into a
        ``PipelineTask``.
    """
    # Build the LLMContext with system prompt + tools bound to env.
    ctx_dict = _build_tools_and_context_lazy(env, task)
    tools = ctx_dict["tools"]
    messages = ctx_dict["messages"]
    # The tau2 policy/instructions must ride along as a system message;
    # without it the agent LLM has no policy and no idea when to stop.
    system = ctx_dict.get("system") or ""
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    context = LLMContext(messages=messages, tools=tools)

    # User aggregator: collects user transcript frames, appends to context,
    # triggers the LLM. Assistant aggregator: streams LLM output into TTS
    # and appends assistant messages back to context.
    user_agg = LLMUserAggregator(context=context)
    assistant_agg = LLMAssistantAggregator(context=context)

    # Apply LLM service settings (some Pipecat versions also require
    # `register_function` style wiring, but FunctionSchema.handler covers
    # the dispatch already).
    try:
        llm_service.register_tools(tools)
    except Exception:
        # Newer Pipecat versions pick up tools from LLMContext automatically.
        pass

    # Stock Pipecat 1.x order: transport.input -> STT -> user_agg -> LLM
    # -> TTS -> transport.output -> assistant_agg.
    #
    # The assistant aggregator is a sink-side observer in 1.x: it consumes
    # text/audio frames for context bookkeeping but does NOT forward LLM
    # text downstream. Placing it before TTS starves TTS (no audio is ever
    # produced). TTS emits AggregatedTextFrame downstream, which the trailing
    # assistant aggregator records into the context.
    pipeline = Pipeline(
        [
            transport.input(),
            _wrap_stt(stt_service, "agent-stt"),
            user_agg,
            llm_service,
            _wrap_tts(tts_service, "agent-tts"),
            transport.output(),
            assistant_agg,
        ]
    )
    logger.info("Built agent pipeline for task=%s domain=%s", task.id, cfg.domain)

    return AgentPipelineParts(pipeline=pipeline, context=context, transport=transport)

"""User-simulator Pipecat pipeline.

Topology:

    transport.input (agent TTS audio) ->  STT  ->  user_agg  ->  user LLM  ->  assistant_agg  ->  TTS  ->  transport.output
                                                                 ^                                            |
                                                                 '----- ###STOP### handled in user_agg ---'

The user pipeline mirrors the agent pipeline but:

- The LLM is the **user simulator** (driven by tau2's ``UserSimulator`` system
  prompt + ``task.user_scenario``).
- No tools are registered. The user LLM does not have any tools — it just
  generates text replies.
- A post-processor watches the assistant aggregator for ``###STOP###``,
  ``###TRANSFER###``, or ``###OUT-OF-SCOPE###`` in the generated text and
  pushes a ``StopTaskFrame`` to teardown both pipelines.

Stop detection is intentionally at the integration boundary (here) rather
than inside tau2's turn-based orchestrator: tau2 is the eval/ground-truth
source, not the runtime driver.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.frames.frames import Frame, LLMFullResponseStartFrame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from pipecat_voice.config import VoiceConfig
from pipecat_voice.transport.virtual_transport import VirtualTransport


def _build_user_system_prompt_lazy(persona_text: str, scenario_text: str) -> str:
    """Lazy import to avoid a circular dep with pipecat_voice.tau2.__init__."""
    from pipecat_voice.tau2.context import build_user_system_prompt

    return build_user_system_prompt(
        persona_text=persona_text, scenario_text=scenario_text
    )


@dataclass
class UserPipelineParts:
    pipeline: BasePipeline
    context: LLMContext
    transport: VirtualTransport
    stop_event: asyncio.Event


class StopOnUserSignalProcessor(FrameProcessor):
    """Sets ``stop_event`` when the user says ###STOP### (or related signals).

    LLM tokens stream in small frames, so a signal token may be split across
    frames (``###ST`` + ``OP###``). We accumulate text within a turn and match
    against the buffer; the buffer resets on each new response turn.
    """

    _SIGNALS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
    _COMPLETION_PHRASES = (
        "that's all i needed",
        "that is all i needed",
        "that's all",
        "thanks, bye",
        "goodbye",
    )

    @classmethod
    def _should_stop(cls, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        return any(signal.lower() in normalized for signal in cls._SIGNALS) or any(
            phrase in normalized for phrase in cls._COMPLETION_PHRASES
        )

    def __init__(self, stop_event: asyncio.Event):
        super().__init__(name="StopOnUserSignal")
        self._stop_event = stop_event
        self._turn_text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._turn_text = ""
        elif isinstance(frame, Frame) and getattr(frame, "text", None):
            self._turn_text += frame.text or ""
            if self._should_stop(self._turn_text):
                logger.info("User simulator emitted a stop or completion signal")
                if not self._stop_event.is_set():
                    self._stop_event.set()
                # Forward the frame downstream so the assistant aggregator
                # still records the message before the pipeline tears down.
        await self.push_frame(frame, direction)


def build_user_pipeline(
    *,
    cfg: VoiceConfig,
    transport: VirtualTransport,
    stt_service: STTService,
    llm_service: Any,
    tts_service: TTSService,
    task,  # tau2 Task
    stop_event: asyncio.Event,
) -> UserPipelineParts:
    """Construct the user simulator's Pipecat pipeline.

    See module docstring for topology. The ``stop_event`` is set when the
    user LLM emits ``###STOP###`` (or related signals); the runner awaits it
    to know when to tear both pipelines down.
    """
    # Build user simulator system prompt from task.user_scenario.
    persona_text = (
        (task.user_scenario.persona or "").strip() if task.user_scenario.persona else ""
    )
    scenario_text = str(task.user_scenario.instructions)
    system_prompt = _build_user_system_prompt_lazy(persona_text, scenario_text)

    # The user side does not have tools. We pass an empty messages list and
    # let the first user-side prompt (from the audio bus) drive the LLM.
    # To bootstrap, seed an initial user message telling the LLM to greet.
    initial_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Begin the call. Greet the agent and state your reason for calling.",
        },
    ]
    context = LLMContext(messages=initial_messages, tools=[])

    user_agg = LLMUserAggregator(context=context)
    assistant_agg = LLMAssistantAggregator(context=context)
    stop_proc = StopOnUserSignalProcessor(stop_event=stop_event)

    # Same stock order as the agent pipeline: the assistant aggregator is
    # a trailing observer (see agent_pipeline). stop_proc sits before TTS so
    # it sees streamed LLM text for ###STOP### detection.
    pipeline = Pipeline(
        [
            transport.input(),
            stt_service,
            user_agg,
            llm_service,
            stop_proc,
            tts_service,
            transport.output(),
            assistant_agg,
        ]
    )
    logger.info("Built user pipeline for task=%s", task.id)
    return UserPipelineParts(
        pipeline=pipeline, context=context, transport=transport, stop_event=stop_event
    )

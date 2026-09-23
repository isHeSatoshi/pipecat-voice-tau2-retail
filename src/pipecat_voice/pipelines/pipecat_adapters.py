"""Pipecat ``STTService``, ``TTSService``, and ``LLMService`` adapters for our protocol impls.

Pipecat's pipeline expects services that emit and consume its own frame
types (``TranscriptionFrame``, ``AudioRawFrame``, ...). Our protocol
implementations (:class:`ParakeetSTT`, :class:`ChatterboxTTS`,
:class:`DummyLLM`, ...) are plain async functions.

These adapters wrap them so they plug into Pipecat's ``Pipeline`` graph
unchanged. They do NOT override any deep behaviour; they only translate
between Pipecat's frame types and our protocol method signatures.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from pipecat_voice.interfaces import LLMResponse


# =============================================================================
# STT adapter
# =============================================================================


class ProtocolSTTService(STTService):
    """Pipecat ``STTService`` that delegates transcription to a protocol impl.

    Utterance buffering: per-chunk inference on 20 ms slices is both
    wasteful (a full model pass per slice) and meaningless (too little
    audio to transcribe). Instead we accumulate ``InputAudioRawFrame``
    audio and transcribe:

    - on ``UserStoppedSpeakingFrame`` (whole utterance, VAD mode),
    - after ~1.2 s without new audio (utterance end, ``--no-vad`` mode), and
    - every ``utterance_window_secs`` of audio as a memory-bound backstop
      (12 s: mid-utterance splits are rare and the novelty gate dedups).

    Empty transcripts are dropped. ``run_stt`` is kept as a single-shot
    entry point for the runner's warm-up path.
    """

    def __init__(
        self,
        *,
        stt_impl,  # pipecat_voice.interfaces.STTProtocol
        sample_rate: int = 16000,
        name: str | None = None,
        utterance_window_secs: float = 12.0,
        **kwargs,
    ):
        # audio_passthrough=False is LOAD-BEARING: forwarding input audio
        # downstream lets it reach VirtualOutputProcessor, which treats any
        # AudioRawFrame as TTS output and pushes it back to the bus — the
        # same chunk then ping-pongs between the two pipelines forever
        # (recirculated 25x realtime in testing). Nothing downstream of STT
        # needs raw input audio.
        kwargs.setdefault("audio_passthrough", False)
        super().__init__(name=name or "ProtocolSTT", **kwargs)
        self._impl = stt_impl
        self._sample_rate = sample_rate
        self._utterance_window_secs = utterance_window_secs
        self._buf = bytearray()
        self._speaking = False
        # Set on the first VAD speech-boundary frame. When the pipeline runs
        # with `--no-vad` no boundaries ever arrive and the user aggregator's
        # turn controller may never fire; flushed transcripts then bypass it
        # via LLMMessagesAppendFrame(run_llm=True) instead.
        self._vad_frames_seen = False
        # Last transcript that triggered an LLM turn (no-VAD mode). Sliding
        # 3 s windows re-transcribe the same audio with small shifts; without
        # this gate every window fires a full LLM turn (feedback storm).
        self._last_trigger_text = ""
        # Gap-flush state: a short utterance may never fill a whole window.
        # A background loop flushes buffered audio after ~1.2 s of silence.
        self._last_append = 0.0
        self._gap_task = None
        # Accounting for flood diagnosis.
        self._audio_in_bytes = 0
        self._audio_in_frames = 0
        self._flush_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
            if self._gap_task is None:
                try:
                    self._gap_task = self.create_task(self._gap_flush_loop())
                except Exception:
                    self._gap_task = asyncio.create_task(self._gap_flush_loop())
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (CancelFrame, EndFrame)):
            await super().process_frame(frame, direction)
            if self._gap_task is not None:
                try:
                    await self.cancel_task(self._gap_task)
                except Exception:
                    pass
                self._gap_task = None
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._speaking = True
            self._vad_frames_seen = True
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._speaking = False
            self._vad_frames_seen = True
            await self._flush_utterance()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InputAudioRawFrame):
            # Bypass the base class's per-chunk run_stt; accumulate instead.
            if frame.audio:
                self._buf.extend(frame.audio)
                self._audio_in_bytes += len(frame.audio)
                self._audio_in_frames += 1
                self._last_append = time.monotonic()
            window_bytes = int(self._sample_rate * 2 * self._utterance_window_secs)
            if len(self._buf) >= window_bytes:
                await self._flush_utterance()
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
            return
        await super().process_frame(frame, direction)

    async def _gap_flush_loop(self) -> None:
        """Flush short utterances that never fill a whole window.

        A 2.5 s TTS reply leaves ~80 KB stranded below the 3 s window;
        without this the conversation stalls after the first exchange.
        """
        import time as _time

        try:
            while True:
                await asyncio.sleep(0.4)
                if (
                    self._buf
                    and self._last_append
                    and (_time.monotonic() - self._last_append) > 1.2
                ):
                    await self._flush_utterance()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"{self.name}: gap loop exited: {e}")

    async def _flush_utterance(self) -> None:
        if not self._buf:
            return
        pcm = bytes(self._buf)
        self._buf.clear()
        self._flush_count += 1
        if self._flush_count % 20 == 1:
            logger.info(
                f"{self.name}: audio_in={self._audio_in_bytes}B "
                f"in {self._audio_in_frames} frames, flushes={self._flush_count}"
            )
        try:
            result = await self._impl.transcribe(pcm, self._sample_rate)
        except Exception as e:
            logger.warning(f"{self.name}: transcribe failed: {e}")
            return
        if not result.text:
            return
        logger.info(f"{self.name}: transcript: {result.text[:160]!r}")
        if not self._vad_frames_seen:
            # No-VAD mode: the turn controller has no speech boundaries to
            # work with, so append the transcript to the context and run the
            # LLM directly. (With VAD, the TranscriptionFrame below feeds the
            # controller, which drives inference at utterance end.)
            #
            # Novelty gate: overlapping windows re-transcribe the same audio
            # with small shifts. Near-duplicates of the last turn trigger are
            # recorded as plain transcriptions (context-visible, no inference)
            # so one utterance yields one LLM turn, not one per window.
            if self._last_trigger_text and _near_duplicate(
                self._last_trigger_text, result.text
            ):
                # Duplicate window: already logged above; dropping keeps the
                # turn controller's accumulator from filling with repeats.
                logger.debug(f"{self.name}: duplicate window dropped")
                return
            self._last_trigger_text = result.text
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": result.text}],
                    run_llm=True,
                )
            )
            return
        if result.is_final:
            await self.push_frame(
                TranscriptionFrame(
                    text=result.text,
                    user_id="",
                    timestamp="",
                    language=None,
                )
            )
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text=result.text,
                    user_id="",
                    timestamp="",
                    language=None,
                )
            )

    async def run_stt(self, audio: bytes) -> AsyncIterator[Frame]:
        result = await self._impl.transcribe(audio, self._sample_rate)
        if not result.text:
            return
        if result.is_final:
            yield TranscriptionFrame(
                text=result.text,
                user_id="",
                timestamp="",
                language=None,
            )
        else:
            yield InterimTranscriptionFrame(
                text=result.text,
                user_id="",
                timestamp="",
                language=None,
            )


# =============================================================================
# TTS adapter
# =============================================================================


class ProtocolTTSService(TTSService):
    """Pipecat ``TTSService`` that delegates ``run_tts`` to a protocol impl.

    Pipecat's TTSService consumes ``LLMTextFrame`` / ``TextFrame`` and emits
    ``AudioRawFrame`` chunks. We hand each text to our protocol's
    ``synthesize`` generator and forward the resulting PCM.
    """

    def __init__(
        self,
        *,
        tts_impl,  # pipecat_voice.interfaces.TTSProtocol
        sample_rate: int = 16000,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, name=name or "ProtocolTTS", **kwargs)
        self._impl = tts_impl
        self._sample_rate = sample_rate

    async def run_tts(self, text: str, context_id: str | None = None) -> AsyncIterator[Frame]:
        async for chunk in self._impl.synthesize(text):
            if not chunk.pcm:
                continue
            # NOTE: must be a real Pipecat Frame. The AudioRawFrame base is
            # a plain mixin; TTSAudioRawFrame is what the pipeline expects.
            yield TTSAudioRawFrame(
                audio=chunk.pcm,
                sample_rate=chunk.sample_rate or self._sample_rate,
                num_channels=1,
            )


# =============================================================================
# LLM adapter
# =============================================================================


class ProtocolLLMService(LLMService):
    """Pipecat ``LLMService`` that delegates to a protocol impl.

    Pipecat's LLMService drives the LLMContextAggregator. For each
    ``LLMContextFrame`` it builds the OpenAI-style message list and calls
    ``run_inference`` (we override that to call our ``complete`` method),
    then emits ``LLMFullResponseStartFrame`` + ``TextFrame`` tokens +
    ``LLMFullResponseEndFrame``.
    """

    def __init__(self, *, llm_impl, name: str | None = None, **kwargs):
        super().__init__(name=name or "ProtocolLLM", **kwargs)
        self._impl = llm_impl

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # LLMService.process_frame does NOT forward frames itself; each
        # subclass is responsible for pushing what it doesn't consume.
        # Without this, StartFrame/CancelFrame die here and the worker
        # stalls in _wait_for_pipeline_start.
        await super().process_frame(frame, direction)
        # Import here to avoid a hard dep at module load.
        from pipecat.frames.frames import LLMContextFrame

        if isinstance(frame, LLMContextFrame):
            await self.process_generator(self.run_inference(frame.context))
        else:
            await self.push_frame(frame, direction)

    async def run_inference(self, context: LLMContext) -> AsyncIterator[Frame]:
        # Convert LLMContext messages -> list of dicts (already in OpenAI
        # shape at runtime; LLMSpecificMessage blobs are skipped).
        messages = [dict(m) for m in context.messages if isinstance(m, dict)]

        # Build tools list (OpenAI shape) for the protocol impl.
        tools = None
        try:
            t = context.tools
            if t is not None and hasattr(t, "standard_tools"):
                tools = [s.to_default_dict() for s in t.standard_tools]
        except Exception:
            tools = None

        # Invoke the protocol.
        response: LLMResponse = await self._impl.complete(
            system="",
            messages=messages,
            tools=tools,
        )

        # Emit start frame.
        yield LLMFullResponseStartFrame()

        if response.tool_calls:
            # Surface tool calls via FunctionCallInProgressFrame + the context.
            from pipecat.adapters.schemas.function_schema import FunctionSchema
            from pipecat.frames.frames import FunctionCallInProgressFrame, FunctionCallResultFrame

            for tc in response.tool_calls:
                yield FunctionCallInProgressFrame(
                    function_name=tc.name,
                    tool_call_id=tc.id,
                    arguments=tc.arguments,
                )
            # Run each handler synchronously here; the schema's handler
            # is responsible for the actual tau2 execution. We invoke them
            # directly and append results back into the context.
            for tc in response.tool_calls:
                schema = _find_schema(context, tc.name)
                result_obj = None
                if schema is not None and getattr(schema, "_handler", None) is not None:
                    try:
                        result_obj = await schema._handler(tc.arguments)
                    except Exception as e:
                        result_obj = {"error": str(e)}
                else:
                    result_obj = {"error": f"unknown tool: {tc.name}"}
                if not isinstance(result_obj, dict):
                    try:
                        result_obj = {"result": str(result_obj)}
                    except Exception:
                        result_obj = {"result": repr(result_obj)}
                yield FunctionCallResultFrame(
                    function_name=tc.name,
                    tool_call_id=tc.id,
                    arguments=tc.arguments,
                    result=result_obj,
                )
        elif response.content:
            # Stream the content as a single TextFrame.
            yield TextFrame(text=response.content)

        yield LLMFullResponseEndFrame()


def _near_duplicate(a: str, b: str, *, threshold: float = 0.6) -> bool:
    """True when two transcripts overlap enough to be the same utterance.

    Sliding STT windows re-emit the same speech with small shifts
    (``"...specific question"`` vs ``"question. It looks like..."``).
    A quick token-overlap check keeps one utterance to one LLM turn.
    """
    import difflib

    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _find_schema(context: LLMContext, name: str):
    try:
        tools = context.tools
        if tools is not None and hasattr(tools, "standard_tools"):
            tools = tools.standard_tools
        for s in tools or []:
            if hasattr(s, "name") and s.name == name:
                return s
            if hasattr(s, "_name") and s._name == name:
                return s
    except Exception:
        return None
    return None

"""In-process virtual transport for the closed-loop Pipecat eval harness.

Two Pipecat pipelines (agent + user simulator) talk to each other over an
in-memory audio bus. Each pipeline has its own ``VirtualTransport`` whose
input is fed from the other pipeline's TTS output and whose output is
captured for the other pipeline's STT input.

Design
------

The transport is split into three concerns:

1. :class:`AudioBus` — two ``asyncio.Queue`` instances (one per direction)
   holding PCM bytes. Public methods ``push`` / ``drain`` are used by the
   runner to bridge the two pipelines without referencing Pipecat directly.

2. :class:`VirtualInputProcessor` — Pipecat ``FrameProcessor`` that reads
   PCM bytes from an ``AudioBus`` source queue and pushes ``AudioRawFrame``
   frames into the pipeline at a configurable frame duration.

3. :class:`VirtualOutputProcessor` — Pipecat ``FrameProcessor`` that
   intercepts outbound ``AudioRawFrame`` frames and forwards them to a
   destination ``AudioBus``.

4. :class:`VirtualTransport` — subclasses ``BaseTransport`` and exposes the
   two processors via ``input()`` / ``output()``. VAD is wired in via the
   standard ``SileroVADAnalyzer`` so the agent pipeline benefits from
   Pipecat's native voice-activity detection and barge-in handling.

Why not just call the services directly?
----------------------------------------

We could feed ``AudioRawFrame`` into the pipelines manually from the runner,
but going through Pipecat's standard transport abstraction gives us:

- VAD via ``SileroVADAnalyzer`` (built-in, well-tested)
- ``UserStartedSpeaking`` / ``UserStoppedSpeaking`` frame emission for
  barge-in propagation
- A clean separation that lets us swap in a real transport (Daily,
  WebRTC, ...) later by replacing only this module
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger
from pydantic import ConfigDict
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams

# 20 ms frames at 16 kHz, 16-bit mono = 640 bytes. Pipecat's transport defaults
# match this; we mirror them so the in-memory pipeline behaves like a real one.
DEFAULT_FRAME_MS = 20
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1


@dataclass
class AudioBus:
    """A two-direction in-memory PCM byte stream.

    Each direction is an unbounded ``asyncio.Queue`` of ``bytes`` chunks. The
    runner feeds one side from the TTS output of the opposite pipeline.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    direction_a_to_b: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)
    direction_b_to_a: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)
    closed: bool = False

    async def push(self, direction: str, pcm: bytes) -> None:
        """Push PCM bytes into one direction of the bus."""
        if self.closed:
            return
        queue = self._queue(direction)
        await queue.put(pcm)

    async def drain(self, direction: str, timeout: float = 0.05) -> bytes:
        """Return one chunk from a direction, blocking up to ``timeout`` seconds.

        Returns ``b""`` on timeout. The VirtualInputProcessor uses this to
        pull PCM in small frame-sized pieces.
        """
        if self.closed:
            return b""
        queue = self._queue(direction)
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return b""

    def _queue(self, direction: str) -> asyncio.Queue[bytes]:
        if direction == "a_to_b":
            return self.direction_a_to_b
        elif direction == "b_to_a":
            return self.direction_b_to_a
        raise ValueError(f"Unknown direction: {direction!r}")


# =============================================================================
# Pipecat processors
# =============================================================================


class VirtualInputProcessor(FrameProcessor):
    """Reads PCM chunks from an :class:`AudioBus` and emits ``InputAudioRawFrame``.

    Each chunk is split into fixed-duration ``InputAudioRawFrame`` frames so
    the pipeline sees audio at a real-time cadence. VAD frames
    (``UserStartedSpeaking`` / ``UserStoppedSpeaking``) are emitted based on
    the VAD analyzer if one is supplied.

    Pipecat 1.x lifecycle note: ``StartFrame`` / ``CancelFrame`` arrive via
    :meth:`process_frame` (``setup()`` is the one-time init hook). Every
    frame — including system frames — must be explicitly pushed downstream,
    otherwise the ``StartFrame`` never reaches the sink and the
    ``PipelineWorker`` stalls in ``_wait_for_pipeline_start``. The audio pump
    is started on ``StartFrame`` and stopped on ``CancelFrame`` / ``EndFrame``
    / ``cleanup()``.
    """

    def __init__(
        self,
        *,
        bus: AudioBus,
        direction: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        frame_ms: int = DEFAULT_FRAME_MS,
        vad_analyzer: Optional[VADAnalyzer] = None,
        name: str | None = None,
    ):
        super().__init__(name=name or f"VirtualInput[{direction}]")
        self._bus = bus
        self._direction = direction
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_bytes = int(sample_rate * channels * 2 * frame_ms / 1000)
        self._vad = vad_analyzer
        self._running = False
        self._vad_state = False  # True = currently speaking
        # Buffers any over-sized chunk into frame-sized pieces.
        self._chunk_buffer = bytearray()
        self._sample_buffer = bytearray()  # for VAD analysis (concatenated PCM)
        self._pump_task = None

    async def setup(self, setup) -> None:
        await super().setup(setup)
        self._running = False
        self._chunk_buffer = bytearray()
        self._sample_buffer = bytearray()
        self._vad_state = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            if not self._running:
                self._running = True
                try:
                    self._pump_task = self.create_task(self._pump())
                except Exception:
                    # Fall back to a bare task if the task manager is unavailable.
                    self._pump_task = asyncio.create_task(self._pump())
            await self.push_frame(frame, direction)
        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._running = False
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        self._running = False
        if self._pump_task is not None:
            try:
                await self.cancel_task(self._pump_task)
            except Exception:
                pass
            self._pump_task = None
        await super().cleanup()

    async def _pump(self) -> None:
        """Background task: pull PCM from the bus, slice into frames, push."""
        try:
            while self._running and not self._bus.closed:
                try:
                    chunk = await self._bus.drain(self._direction, timeout=0.02)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug(f"{self.name}: bus drain failed: {e}")
                    continue
                if not chunk:
                    # Idle tick: feed silence to VAD so it can decay to QUIET
                    # after speech ends (the bus delivers nothing when idle).
                    await self._maybe_vad_tick(self._silence_tick(), idle=True)
                    continue

                self._chunk_buffer.extend(chunk)
                while len(self._chunk_buffer) >= self._frame_bytes:
                    frame_bytes = bytes(self._chunk_buffer[: self._frame_bytes])
                    del self._chunk_buffer[: self._frame_bytes]
                    await self._emit_frame(frame_bytes)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"{self.name}: pump exited: {e}")
        finally:
            # Final flush so VAD can mark stop.
            if self._vad_state:
                try:
                    await self._emit_speaking_frame(False)
                except Exception:
                    pass
                self._vad_state = False

    def _silence_tick(self) -> bytes:
        """One frame-sized chunk of silence for idle VAD ticks."""
        return b"\x00" * self._frame_bytes

    async def _emit_frame(self, pcm: bytes) -> None:
        # NOTE: must be InputAudioRawFrame (a real Pipecat Frame and the type
        # STTService listens for). The AudioRawFrame base is a plain mixin.
        frame = InputAudioRawFrame(
            audio=pcm, sample_rate=self._sample_rate, num_channels=self._channels
        )
        try:
            await self.push_frame(frame)
        except Exception as e:
            logger.debug(f"{self.name}: push audio failed: {e}")
        await self._maybe_vad_tick(pcm)

    async def _maybe_vad_tick(self, pcm: bytes, idle: bool = False) -> None:
        """Run VAD on the accumulated audio and emit start/stop frames."""
        if self._vad is None:
            return
        # When idle and not speaking there is nothing to decay; skip work.
        if idle and not self._vad_state and not self._sample_buffer:
            return
        self._sample_buffer.extend(pcm)
        # Run VAD at ~30 ms windows; Pipecat's Silero expects 16 kHz int16 mono.
        window = self._vad.sample_rate // 33 * 2
        while len(self._sample_buffer) >= window:
            audio_window = bytes(self._sample_buffer[:window])
            del self._sample_buffer[:window]
            try:
                state = await self._vad.analyze_audio(audio_window)
            except Exception as e:
                logger.debug(f"{self.name}: VAD failed: {e}")
                return
            # Only SPEAKING/QUIET are stable; STARTING/STOPPING are transitional.
            if state == VADState.SPEAKING and not self._vad_state:
                await self._emit_speaking_frame(True)
                self._vad_state = True
            elif state == VADState.QUIET and self._vad_state:
                await self._emit_speaking_frame(False)
                self._vad_state = False

    async def _emit_speaking_frame(self, speaking: bool) -> None:
        frame_cls = UserStartedSpeakingFrame if speaking else UserStoppedSpeakingFrame
        await self.push_frame(frame_cls(), direction=FrameDirection.DOWNSTREAM)


class VirtualOutputProcessor(FrameProcessor):
    """Captures outbound TTS audio frames and forwards them to the bus.

    Only ``OutputAudioRawFrame`` / ``TTSAudioRawFrame`` are captured.
    ``InputAudioRawFrame`` is explicitly excluded even though it subclasses
    the ``AudioRawFrame`` mixin: if input audio ever leaks downstream it must
    NOT be re-injected into the bus (that recirculates the same chunk between
    both pipelines without bound).
    """

    def __init__(
        self,
        *,
        bus: AudioBus,
        direction: str,
        name: str | None = None,
    ):
        super().__init__(name=name or f"VirtualOutput[{direction}]")
        self._bus = bus
        self._direction = direction
        self._buffer = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame))
            and not isinstance(frame, InputAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            self._buffer.extend(frame.audio)
            # Flush in ~20 ms slices so the bus does not see megabyte chunks.
            slice_bytes = frame.sample_rate * frame.num_channels * 2 * 20 // 1000
            while len(self._buffer) >= slice_bytes:
                chunk = bytes(self._buffer[:slice_bytes])
                del self._buffer[:slice_bytes]
                await self._bus.push(self._direction, chunk)
        await self.push_frame(frame, direction)


# =============================================================================
# Transport
# =============================================================================


class VirtualTransportParams(TransportParams):
    """Transport parameters for the in-process virtual transport.

    Inherits from Pipecat's Pydantic ``TransportParams`` and adds our
    in-process-only fields: ``bus``, ``direction``, ``vad_analyzer``,
    ``sample_rate``, ``channels``, ``frame_ms``.

    Pipecat 1.x expects ``audio_in_sample_rate`` / ``audio_out_sample_rate``
    to match the ``sample_rate`` we use on the bus, so we set them as
    defaults in the same model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bus: AudioBus = AudioBus()
    direction: str = "a_to_b"
    vad_analyzer: Optional[VADAnalyzer] = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    audio_in_sample_rate: int = DEFAULT_SAMPLE_RATE
    audio_out_sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    frame_ms: int = DEFAULT_FRAME_MS


class VirtualTransport(BaseTransport):
    """A Pipecat transport whose audio I/O is wired through an :class:`AudioBus`.

    ``direction`` is the direction of audio coming INTO this transport's
    pipeline. The transport's output is the opposite direction, automatically
    fed back into the bus for the other side to consume.
    """

    def __init__(self, *, params: VirtualTransportParams, name: str | None = None):
        super().__init__(
            name=name or f"VirtualTransport[{params.direction}]",
            input_name=f"VirtualInput[{params.direction}]",
            output_name=f"VirtualOutput[{params.direction}]",
        )
        self._params = params
        self._input: Optional[VirtualInputProcessor] = None
        self._output: Optional[VirtualOutputProcessor] = None

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = VirtualInputProcessor(
                bus=self._params.bus,
                direction=self._params.direction,
                sample_rate=self._params.sample_rate,
                channels=self._params.channels,
                frame_ms=self._params.frame_ms,
                vad_analyzer=self._params.vad_analyzer,
                name=self._input_name,
            )
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            # The opposite direction: if we read from a_to_b, we write into b_to_a.
            opposite = "b_to_a" if self._params.direction == "a_to_b" else "a_to_b"
            self._output = VirtualOutputProcessor(
                bus=self._params.bus,
                direction=opposite,
                name=self._output_name,
            )
        return self._output


def make_default_vad(sample_rate: int = DEFAULT_SAMPLE_RATE) -> SileroVADAnalyzer:
    """Build a Silero VAD analyzer at the given sample rate.

    ``SileroVADAnalyzer`` is the standard Pipecat VAD, used by every official
    transport. Defaults are tuned for 16 kHz mono.
    """
    return SileroVADAnalyzer(sample_rate=sample_rate)

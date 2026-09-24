from __future__ import annotations

from typing import Any

from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver

from pipecat_voice.observability.trace_writer import TraceWriter


class VoiceTraceObserver(BaseObserver):
    def __init__(self, trace: TraceWriter, side: str):
        super().__init__(name=f"VoiceTraceObserver[{side}]")
        self._trace = trace
        self._side = side
        self._seen: set[str] = set()
        self._errors: list[str] = []

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    async def on_push_frame(self, data) -> None:
        frame = data.frame
        payload = (
            getattr(frame, "tool_call_id", None),
            getattr(frame, "function_name", None),
            getattr(frame, "text", None),
        )
        key = f"{self._side}:{type(frame).__name__}:{data.timestamp}:{payload}"
        if key in self._seen:
            return
        self._seen.add(key)
        event: dict[str, Any] | None = None
        if isinstance(frame, UserStartedSpeakingFrame):
            event = {"type": "vad_start", "side": self._side}
        elif isinstance(frame, UserStoppedSpeakingFrame):
            event = {"type": "vad_stop", "side": self._side}
        elif isinstance(frame, TranscriptionFrame):
            event = {
                "type": "stt_result",
                "side": self._side,
                "text": frame.text,
                "user_id": frame.user_id,
                "timestamp": frame.timestamp,
            }
        elif isinstance(frame, LLMRunFrame):
            event = {"type": "llm_run", "side": self._side}
        elif isinstance(frame, LLMFullResponseStartFrame):
            event = {"type": "llm_start", "side": self._side}
        elif isinstance(frame, LLMFullResponseEndFrame):
            event = {"type": "llm_end", "side": self._side}
        elif isinstance(frame, FunctionCallInProgressFrame):
            event = {
                "type": "tool_call",
                "side": self._side,
                "name": frame.function_name,
                "tool_call_id": frame.tool_call_id,
                "arguments": frame.arguments,
            }
        elif isinstance(frame, FunctionCallResultFrame):
            event = {
                "type": "tool_result",
                "side": self._side,
                "name": frame.function_name,
                "tool_call_id": frame.tool_call_id,
                "result": frame.result,
            }
        elif isinstance(frame, TTSStartedFrame):
            event = {"type": "tts_start", "side": self._side}
        elif isinstance(frame, TTSStoppedFrame):
            event = {"type": "tts_stop", "side": self._side}
        elif isinstance(frame, ErrorFrame):
            error = str(getattr(frame, "error", frame))
            self._errors.append(error)
            event = {
                "type": "frame_error",
                "side": self._side,
                "error": error,
            }
        if event is not None:
            self._trace.emit(event)

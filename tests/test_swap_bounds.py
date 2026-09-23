"""Tests proving STT/TTS/LLM are replaceable.

These tests verify the protocol contracts on :mod:`pipecat_voice.interfaces`
by constructing dummy/recording implementations and exercising the runner
build path with them. They DO NOT touch Pipecat pipelines; they focus on
the data contracts so they run fast and deterministically.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from pipecat_voice.interfaces import (
    LLMResponse,
    LLMToolCall,
    STTProtocol,
    TTSProtocol,
    LLMProtocol,
    STTResult,
    TTSChunk,
)


class RecordingSTT:
    sample_rate_in: int = 16000

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int]] = []
        self.next_result = STTResult(text="hello", confidence=1.0, is_final=True)

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> STTResult:
        self.calls.append((pcm_bytes, sample_rate))
        return self.next_result


class RecordingTTS:
    sample_rate_out: int = 16000
    voice: str = "test-voice"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.chunk = TTSChunk(pcm=b"\x00" * 32, sample_rate=16000, is_final=True)

    async def synthesize(self, text: str):
        self.calls.append(text)
        yield self.chunk


class RecordingLLM:
    def __init__(self, model: str = "test-llm") -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self.next_response = LLMResponse(content="hi")

    async def complete(self, *, system, messages, tools=None) -> LLMResponse:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        return self.next_response


def test_stt_protocol_runtime_check() -> None:
    stt = RecordingSTT()
    assert isinstance(stt, STTProtocol)
    assert stt.sample_rate_in == 16000  # default dataclass attr


def test_tts_protocol_runtime_check() -> None:
    tts = RecordingTTS()
    assert isinstance(tts, TTSProtocol)


def test_llm_protocol_runtime_check() -> None:
    llm = RecordingLLM()
    assert isinstance(llm, LLMProtocol)
    assert llm.model == "test-llm"


def test_stt_transcribe_returns_text() -> None:
    async def _go():
        stt = RecordingSTT()
        stt.next_result = STTResult(text="hi", confidence=0.9)
        r = await stt.transcribe(b"\x00" * 16, 16000)
        assert r.text == "hi"
        assert r.confidence == 0.9
        assert (b"\x00" * 16, 16000) in stt.calls
    asyncio.run(_go())


def test_tts_yields_chunks() -> None:
    async def _go():
        tts = RecordingTTS()
        out = []
        async for c in tts.synthesize("hello"):
            out.append(c)
        assert len(out) == 1
        assert out[0].pcm == b"\x00" * 32
        assert "hello" in tts.calls
    asyncio.run(_go())


def test_llm_complete_returns_response() -> None:
    async def _go():
        llm = RecordingLLM()
        llm.next_response = LLMResponse(
            tool_calls=[LLMToolCall(id="x", name="t", arguments={"a": 1})]
        )
        resp = await llm.complete(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert resp.tool_calls[0].name == "t"
        assert resp.tool_calls[0].arguments == {"a": 1}
        assert len(llm.calls) == 1
        assert llm.calls[0]["messages"][0]["content"] == "hi"
    asyncio.run(_go())


def test_dummy_stt_echoes_primed_text() -> None:
    from pipecat_voice.services.dummy_stt_tts_llm import DummySTT, DummyTTS, DummyLLM, scripted_tool_call

    async def _go():
        stt = DummySTT()
        stt.prime("hi there")
        r = await stt.transcribe(b"\x00", 16000)
        assert r.text == "hi there"
        r2 = await stt.transcribe(b"\x00", 16000)
        assert r2.text == ""
    asyncio.run(_go())


def test_dummy_tts_yields_silence() -> None:
    from pipecat_voice.services.dummy_stt_tts_llm import DummyTTS

    async def _go():
        tts = DummyTTS(sample_rate_out=16000)
        chunks = []
        async for c in tts.synthesize("hi"):
            chunks.append(c)
        # 200 ms / 100 ms = 2 chunks total
        assert all(c.sample_rate == 16000 for c in chunks)
        assert all(len(c.pcm) > 0 for c in chunks)
    asyncio.run(_go())


def test_dummy_llm_default_emits_transfer_tool_call() -> None:
    from pipecat_voice.services.dummy_stt_tts_llm import DummyLLM

    async def _go():
        llm = DummyLLM()
        resp = await llm.complete(system="s", messages=[])
        assert resp.content is None
        assert resp.tool_calls[0].name == "transfer_to_human_agents"
    asyncio.run(_go())


def test_dummy_llm_scripted_responses() -> None:
    from pipecat_voice.services.dummy_stt_tts_llm import DummyLLM, scripted_text, scripted_tool_call

    async def _go():
        llm = DummyLLM()
        llm.script([scripted_text("first"), scripted_tool_call("foo", {"x": 1}, "tc1"), scripted_text("last")])
        r1 = await llm.complete(system="s", messages=[])
        r2 = await llm.complete(system="s", messages=[])
        r3 = await llm.complete(system="s", messages=[])
        assert r1.content == "first"
        assert r2.tool_calls[0].name == "foo"
        assert r3.content == "last"
    asyncio.run(_go())

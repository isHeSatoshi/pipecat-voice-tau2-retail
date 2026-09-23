"""Minimal pipeline probe with real Anthropic + dummy STT/TTS.

This script builds the simplest possible Pipecat pipeline that uses the
AnthropicLLMService (with a MiniMax-compatible endpoint) and tries to run
one LLM call. It exists to isolate whether the StartFrame timeout we see
in the full harness comes from the LLM service itself or from the wider
pipeline composition.
"""
from __future__ import annotations

import asyncio

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.services.anthropic.llm import AnthropicLLMService, AnthropicLLMSettings
from pipecat.services.tts_service import TTSService
from anthropic import AsyncAnthropic


async def main():
    cfg_url = "https://api.minimax.io/anthropic"
    api_key = ""  # filled from env
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    client = AsyncAnthropic(api_key=api_key, base_url=cfg_url)
    settings = AnthropicLLMSettings(model="MiniMax-M2.7-High-Speed", max_tokens=128, temperature=0.0)
    llm = AnthropicLLMService(api_key=api_key, settings=settings, client=client)

    # Minimal context.
    ctx = LLMContext(messages=[
        {"role": "system", "content": "Reply with a single short sentence."},
        {"role": "user", "content": "Say hi."},
    ])
    user_agg = LLMUserAggregator(context=ctx)
    assistant_agg = LLMAssistantAggregator(context=ctx)

    pipeline = Pipeline([user_agg, llm, assistant_agg])
    worker = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True))

    runner = PipelineRunner()
    print("Running minimal Anthropic pipeline...")
    await asyncio.wait_for(runner.run(worker), timeout=30)
    print("Messages in context:", ctx.messages)
    for m in ctx.messages:
        if m.get("role") == "assistant" and m.get("content"):
            print("Assistant:", m["content"][:120])


if __name__ == "__main__":
    asyncio.run(main())

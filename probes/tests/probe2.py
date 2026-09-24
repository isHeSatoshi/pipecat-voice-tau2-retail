"""Probe 2: full pipeline (transport + STT/TTS adapters + Anthropic LLM)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Direct submodule imports to avoid pipecat_voice/__init__.py side effects.
from pipecat_voice.services.dummy_stt_tts_llm import DummySTT, DummyTTS
from pipecat_voice.pipelines.pipecat_adapters import ProtocolSTTService, ProtocolTTSService
from pipecat_voice.transport.virtual_transport import AudioBus, VirtualTransport, VirtualTransportParams

import asyncio
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator, LLMUserAggregator,
)
from pipecat.services.anthropic.llm import AnthropicLLMService, AnthropicLLMSettings
from anthropic import AsyncAnthropic


async def go():
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    client = AsyncAnthropic(api_key=api_key, base_url='https://api.minimax.io/anthropic')
    settings = AnthropicLLMSettings(model='MiniMax-M2.7-High-Speed', max_tokens=128, temperature=0.0)
    llm = AnthropicLLMService(api_key=api_key, settings=settings, client=client)

    ctx = LLMContext(messages=[
        {'role': 'system', 'content': 'Reply with one short sentence.'},
        {'role': 'user', 'content': 'Say hi.'},
    ])
    user_agg = LLMUserAggregator(context=ctx)
    assistant_agg = LLMAssistantAggregator(context=ctx)
    stt = ProtocolSTTService(stt_impl=DummySTT(sample_rate_in=16000), sample_rate=16000)
    tts = ProtocolTTSService(tts_impl=DummyTTS(sample_rate_out=16000), sample_rate=16000)
    bus = AudioBus(sample_rate=16000)
    transport = VirtualTransport(params=VirtualTransportParams(
        bus=bus, direction='b_to_a', vad_analyzer=None, sample_rate=16000,
    ))

    pipeline = Pipeline([
        transport.input(), stt, user_agg, llm, assistant_agg, tts, transport.output(),
    ])
    worker = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    async def kick():
        await asyncio.sleep(2)
        await worker.queue_frame(LLMRunFrame())
        print('KICK')

    await asyncio.wait_for(asyncio.gather(runner.run(worker), kick()), timeout=30)
    print('DONE')
    for m in ctx.messages:
        if m.get('content'):
            print(f'  [{m["role"]}] {str(m["content"])[:200]}')


asyncio.run(go())

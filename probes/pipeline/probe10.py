"""Probe 10: TWO pipelines with real Anthropic LLM."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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

    bus = AudioBus(sample_rate=16000)
    vad = None

    def make_pipeline(direction: str):
        llm = AnthropicLLMService(api_key=api_key, settings=settings, client=client)
        ctx = LLMContext(messages=[
            {'role': 'system', 'content': 'Reply with one short sentence.'},
            {'role': 'user', 'content': 'hi.'},
        ])
        user_agg = LLMUserAggregator(context=ctx)
        assistant_agg = LLMAssistantAggregator(context=ctx)
        stt = ProtocolSTTService(stt_impl=DummySTT(sample_rate_in=16000), sample_rate=16000)
        tts = ProtocolTTSService(tts_impl=DummyTTS(sample_rate_out=16000), sample_rate=16000)
        transport = VirtualTransport(params=VirtualTransportParams(
            bus=bus, direction=direction, vad_analyzer=vad, sample_rate=16000,
        ))
        return Pipeline([
            transport.input(), stt, user_agg, llm, assistant_agg, tts, transport.output(),
        ]), ctx

    agent_pipeline, agent_ctx = make_pipeline('b_to_a')
    user_pipeline, user_ctx = make_pipeline('a_to_b')

    agent_worker = PipelineWorker(agent_pipeline, params=PipelineParams(allow_interruptions=True), start_timeout_secs=240, name='agent')
    user_worker = PipelineWorker(user_pipeline, params=PipelineParams(allow_interruptions=True), start_timeout_secs=240, name='user')

    runner = PipelineRunner()
    await runner.add_workers(agent_worker, user_worker)

    async def stopper():
        await asyncio.sleep(30)
        await runner.cancel()

    async def kicks():
        await asyncio.sleep(2)
        await user_worker.queue_frame(LLMRunFrame())
        await agent_worker.queue_frame(LLMRunFrame())

    await asyncio.wait_for(asyncio.gather(runner.run(), stopper(), kicks()), timeout=300)
    print('TWO-PIPELINE ANTHROPIC DONE')
    for n, ctx in [('agent', agent_ctx), ('user', user_ctx)]:
        print(f'--- {n} messages ---')
        for m in ctx.messages:
            if m.get('content'):
                print(f'  [{m["role"]}] {str(m["content"])[:200]}')


asyncio.run(go())

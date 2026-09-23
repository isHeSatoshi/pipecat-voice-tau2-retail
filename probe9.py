"""Probe 9: TWO pipelines like the runner does, with DummyLLM."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipecat_voice.services.dummy_stt_tts_llm import DummySTT, DummyTTS, DummyLLM
from pipecat_voice.pipelines.pipecat_adapters import ProtocolSTTService, ProtocolTTSService, ProtocolLLMService
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


async def go():
    bus = AudioBus(sample_rate=16000)
    vad = None

    def make_pipeline(direction: str):
        llm = ProtocolLLMService(llm_impl=DummyLLM(), name=f'llm-{direction}')
        ctx = LLMContext(messages=[
            {'role': 'system', 'content': 'reply ok'},
            {'role': 'user', 'content': 'hi'},
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

    agent_worker = PipelineWorker(agent_pipeline, params=PipelineParams(allow_interruptions=True), start_timeout_secs=180, name='agent')
    user_worker = PipelineWorker(user_pipeline, params=PipelineParams(allow_interruptions=True), start_timeout_secs=180, name='user')

    runner = PipelineRunner()
    await runner.add_workers(agent_worker, user_worker)

    async def stopper():
        await asyncio.sleep(20)
        await runner.cancel()

    async def kicks():
        await asyncio.sleep(2)
        await user_worker.queue_frame(LLMRunFrame())
        await agent_worker.queue_frame(LLMRunFrame())

    await asyncio.wait_for(asyncio.gather(runner.run(), stopper(), kicks()), timeout=200)
    print('two-pipeline DONE')


asyncio.run(go())

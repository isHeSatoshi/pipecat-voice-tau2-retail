"""Probe 8: full chain with Dummy LLM via ProtocolLLMService."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipecat_voice.services.dummy_stt_tts_llm import DummySTT, DummyTTS, DummyLLM
from pipecat_voice.pipelines.pipecat_adapters import ProtocolSTTService, ProtocolTTSService, ProtocolLLMService
from pipecat_voice.transport.virtual_transport import AudioBus, VirtualTransport, VirtualTransportParams

import asyncio
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator, LLMUserAggregator


async def go():
    llm = ProtocolLLMService(llm_impl=DummyLLM(), name='dummy-llm')
    ctx = LLMContext(messages=[{'role':'system','content':'reply ok'},{'role':'user','content':'hi'}])
    user_agg = LLMUserAggregator(context=ctx)
    assistant_agg = LLMAssistantAggregator(context=ctx)
    stt = ProtocolSTTService(stt_impl=DummySTT(sample_rate_in=16000), sample_rate=16000)
    tts = ProtocolTTSService(tts_impl=DummyTTS(sample_rate_out=16000), sample_rate=16000)
    bus = AudioBus(sample_rate=16000)
    transport = VirtualTransport(params=VirtualTransportParams(bus=bus, direction='b_to_a', vad_analyzer=None, sample_rate=16000))
    pipeline = Pipeline([transport.input(), stt, user_agg, llm, assistant_agg, tts, transport.output()])
    worker = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True), start_timeout_secs=180)
    runner = PipelineRunner()
    await asyncio.wait_for(runner.run(worker), timeout=200)
    print('dummy-LLM DONE')


asyncio.run(go())

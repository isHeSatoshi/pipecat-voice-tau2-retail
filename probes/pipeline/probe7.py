"""Probe 7: bisect the LLM block — remove assistant_agg."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipecat_voice.services.dummy_stt_tts_llm import DummySTT, DummyTTS
from pipecat_voice.pipelines.pipecat_adapters import ProtocolSTTService, ProtocolTTSService
from pipecat_voice.transport.virtual_transport import AudioBus, VirtualTransport, VirtualTransportParams

import asyncio
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.services.anthropic.llm import AnthropicLLMService, AnthropicLLMSettings
from anthropic import AsyncAnthropic


async def go():
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    client = AsyncAnthropic(api_key=api_key, base_url='https://api.minimax.io/anthropic')
    settings = AnthropicLLMSettings(model='MiniMax-M2.7-High-Speed', max_tokens=128, temperature=0.0)
    llm = AnthropicLLMService(api_key=api_key, settings=settings, client=client)
    ctx = LLMContext(messages=[{'role':'system','content':'hi'},{'role':'user','content':'reply ok'}])
    user_agg = LLMUserAggregator(context=ctx)
    stt = ProtocolSTTService(stt_impl=DummySTT(sample_rate_in=16000), sample_rate=16000)
    tts = ProtocolTTSService(tts_impl=DummyTTS(sample_rate_out=16000), sample_rate=16000)
    bus = AudioBus(sample_rate=16000)
    transport = VirtualTransport(params=VirtualTransportParams(bus=bus, direction='b_to_a', vad_analyzer=None, sample_rate=16000))
    # Test: STT → LLM (no assistant_agg, no TTS)
    pipeline = Pipeline([transport.input(), stt, user_agg, llm, tts, transport.output()])
    worker = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True), start_timeout_secs=120)
    runner = PipelineRunner()
    await asyncio.wait_for(runner.run(worker), timeout=130)
    print('no-assistant DONE')


asyncio.run(go())

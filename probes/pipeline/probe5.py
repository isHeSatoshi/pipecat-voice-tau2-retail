"""Probe 5: transport + STT + TTS (no LLM)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipecat_voice.services.dummy_stt_tts_llm import DummySTT, DummyTTS
from pipecat_voice.pipelines.pipecat_adapters import ProtocolSTTService, ProtocolTTSService
from pipecat_voice.transport.virtual_transport import AudioBus, VirtualTransport, VirtualTransportParams

import asyncio
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker


async def go():
    bus = AudioBus(sample_rate=16000)
    transport = VirtualTransport(params=VirtualTransportParams(
        bus=bus, direction='b_to_a', vad_analyzer=None, sample_rate=16000,
    ))
    stt = ProtocolSTTService(stt_impl=DummySTT(sample_rate_in=16000), sample_rate=16000)
    tts = ProtocolTTSService(tts_impl=DummyTTS(sample_rate_out=16000), sample_rate=16000)
    pipeline = Pipeline([transport.input(), stt, tts, transport.output()])
    worker = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()
    await asyncio.wait_for(runner.run(worker), timeout=15)
    print('transport+stt+tts DONE')


asyncio.run(go())

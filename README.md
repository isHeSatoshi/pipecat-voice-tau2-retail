# Pipecat Voice + Tau-bench Retail Evals

## A voice agent you can hear, inspect, and argue with

This project is a real-time retail customer-service agent built with Pipecat and evaluated against Tau-bench retail tasks. It is not a transcript mockup. A simulated customer speaks, Parakeet transcribes the audio, the agent reasons over grounded retail tools, Chatterbox speaks the reply, and the resulting database state is scored against the task.

The important part is not only the final answer. Every run keeps the conversation, tool calls, trace events, audio channels, reward breakdown, and failure checks so a bad result can be understood instead of quietly discarded.

The current bundle contains the iterative `v4_calibrated_batch` development history. It includes successful runs, failed runs, retries, recovered transcripts, and test runs. That history is intentional: it shows how the agent changed as the problems became clearer.

## What is in the result bundle

The derived report is built from the retained run artifacts without changing the raw trajectories, traces, or audio:

- [report analysis](docs/report/analysis.md)
- [task summary table](docs/report/task_summary.csv)
- [attempt-level results](docs/report/attempts.csv)
- [behavior checks](docs/report/checks.csv)
- [artifact inventory](docs/report/artifacts.csv)
- [provenance and caveats](docs/report/provenance.json)

### Task outcomes

![Task outcomes](docs/report/charts/task_outcomes.png)

### Every recorded attempt

![Attempt outcomes](docs/report/charts/attempt_outcomes.png)

### Behavior-check pass rates

![Behavior check pass rates](docs/report/charts/check_pass_rates.png)

### Attempt duration

![Attempt durations](docs/report/charts/attempt_durations.png)

### Representative check matrix

![Representative check matrix](docs/report/charts/representative_check_matrix.png)

## A real conversation to listen to

The representative conversation below is task 7, simulation `sim_27c7509b`. It reached local reward `1.0`, matched the database state, and completed all `6/6` recorded actions. The two channels are kept separate so the customer voice and agent voice do not get confused with one another.

- [agent transcript](docs/report/conversation_agent_task7.txt)
- [customer transcript](docs/report/conversation_user_task7.txt)
- [agent audio](docs/report/audio/conversation_agent_task7.wav)
- [customer audio](docs/report/audio/conversation_user_task7.wav)
- [mixed reference conversation](data/runs/v4_calibrated_batch/task_7/sim_27c7509b/conversation.wav)

The separate audio files are the useful files for checking what each side actually said. The mixed reference file is intentionally mixed and should not be used to judge speaker-specific alignment.

## The short version of the results

The report contains 19 recorded attempts across 10 unique task ids. The bundle mixes the development tasks 0–7 with the selected test tasks 5, 9, and 12, and the root summary records both train and test provenance. That mixed provenance is preserved and called out rather than hidden.

- `9/19` attempts reached local reward `1.0` (`47.4%`).
- `9/19` attempts matched the expected database state (`47.4%`).
- `6/19` attempts timed out (`31.6%`).
- Only `3/19` attempts passed all three behavior checks.
- Strict Tau2 reward was unavailable for all 19 records because the NL judge was not configured.
- Task 5 is a clear failure: all four recorded attempts scored local reward `0.0`.
- Task 9 is a qualified local success: database and actions matched, but one recorded authentication call returned `User not found`.
- Task 12 reached local reward `1.0`, but the write used the wrong payment method. The database result and the write-protocol check disagree, which is exactly why both are shown.

The full attempt history is in [attempts.csv](docs/report/attempts.csv). The representative task table is in [task_summary.csv](docs/report/task_summary.csv).

## The system I built

```text
customer LLM
    ↓
customer TTS ───────────────┐
    ↓                        │
virtual PCM audio bus        │
    ↓                        │
Parakeet STT + Silero VAD    │
    ↓                        │
customer-service agent LLM   │
    ↓                        │
grounded Tau-bench tools     │
    ↓                        │
exchange / return / policy   │
    ↓                        │
Chatterbox TTS ─────────────┘
    ↓
next customer turn
```

Pipecat owns the streaming pipeline. Tau-bench owns the retail environment, user scenarios, tools, database state, and evaluation model. The virtual transport lets both sides run in one process while still passing real PCM audio through STT, VAD, LLM, TTS, and tool boundaries.

### Models and services

- NVIDIA Parakeet TDT 0.6B v3 for speech-to-text.
- Chatterbox for text-to-speech.
- Silero VAD for speech boundaries and turn-taking.
- MiniMax M2.7 for the customer-service agent and the simulated customer.
- Pipecat 1.11 for the voice pipeline.
- Tau-bench retail for the environment, tools, tasks, and evaluation.

## Why the design looks like this

### Cascaded speech instead of a native audio model

The project deliberately uses a cascaded `stt → llm → tts` path. That makes the behavior easy to inspect: I can see the transcription, the reasoning context, the tool arguments, the tool result, and the generated speech. A native audio model may have lower latency, but it would make debugging much less transparent.

### One virtual audio bus, two independent voices

The agent and user simulator are not two unrelated recordings. Their TTS output is routed through a two-direction PCM bus. Parakeet receives the opposite side’s audio, Silero decides when a turn starts and stops, and the resulting audio is written to separate agent and user files.

This design also exposed a real class of bugs: audio feedback loops, premature STT flushes, fragmented turns, repeated checking phrases, and transcript/audio misalignment. Those are problems a static prompt cannot solve.

### Grounded Tau-bench tools

The agent does not invent order ids, item ids, prices, payment methods, or exchange results. It reads them from the Tau-bench environment. Tool schemas are generated from the actual Tau-bench tool parameters, and the runtime validates required and unknown arguments before the environment can mutate.

### A state-machine prompt, not a wall of prohibitions

Prompt `v4` is organized around the actual conversation state:

1. authenticate the customer;
2. locate the correct order;
3. read the current items and product variants;
4. resolve ordered preferences;
5. state the exact write;
6. wait for an immediate explicit confirmation;
7. execute one write;
8. report the real result;
9. end naturally.

The agent also follows the latest request. If the customer changes an exchange before confirming it, the old proposal is discarded rather than executed accidentally.

### Runtime safety around writes

Prompt instructions are helpful, but they are not a transaction lock. The runtime adds guards for exact repeated tool calls, malformed arguments, and write ordering. A retail write must follow a proposal and an immediate affirmative response. This is why the project can show both the model’s reasoning and the safety boundary around the tool call.

## The debugging story

The baseline was not merely “a bad prompt.” It exposed several interacting failures:

- Silero VAD was not initialized consistently, so speech boundaries were unreliable.
- Tool schemas exposed the wrong metadata, which led to empty or incorrect arguments.
- Repeated tool calls were executed instead of being stopped safely.
- TTS sentence gaps looked like completed customer turns, flooding the context.
- Names were misspelled by STT, especially first and last names.
- Zip codes were spoken as a block instead of digit by digit, making recovery harder.
- The agent sometimes ignored an ordered preference such as `battery → USB → AC`.
- Holding phrases such as “checking, just a sec” were treated as new customer turns.
- The original audio artifact builder estimated clip boundaries from wall-clock timing, which could attach the wrong voice to a transcript line.

The fixes were deliberately generic. The system asks for a first name and last name one letter at a time, asks for a zip code digit by digit, retries authentication after a clear spelling turn, interprets ordered preferences literally, waits for confirmation, and ignores non-substantive hold phrases. No task-specific customer identity or product id was hardcoded into the agent.

New runs capture exact per-turn agent and user audio. Legacy `v4_calibrated_batch` artifacts are preserved; unsafe legacy per-message links are disabled through sidecar manifests rather than guessed.

## Run it

The project expects Python 3.12 and a configured model key. The Tau-bench runtime is provided by the project environment and declared dependency set used by this checkout.

```powershell
cd D:\Project\infer_task\pipecat_voice
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,viewer]"
copy .env.example .env
```

Set the model credentials in `.env`:

```text
MINIMAX_API_KEY=...
```

The default MiniMax endpoint is `https://api.minimax.io/anthropic`.

### Offline smoke test

```powershell
python -m pipecat_voice.cli run --domain retail --task 0 `
  --stt dummy --tts dummy --agent-llm dummy --user-llm dummy `
  --max-seconds 8 --out data/runs/dummy_smoke
```

### Real-stack run

```powershell
python -m pipecat_voice.cli run --domain retail --task 7 `
  --num-trials 1 --max-seconds 480 --prompt-variant v4 `
  --agent-llm minimax --user-llm minimax `
  --stt parakeet --tts chatterbox --seed 42 `
  --out data/runs/v4_new_task --check all
```

### Analyze saved runs without models

```powershell
python -m pipecat_voice.cli analyze --run data/runs/v4_calibrated_batch
```

## Open the dashboard

```powershell
streamlit run viewer/app.py
```

The viewer lets me select a run and a simulation, then inspect:

- the role-aware transcript;
- exact tool names and arguments;
- reward, database match, and action matches;
- termination reason and duration;
- the event timeline;
- separate agent and customer audio;
- run comparison and retained failed attempts.

The dashboard is deliberately a debugging surface, not just a leaderboard. The trace timeline is often more useful than the final score because it shows when the agent, user simulator, STT, VAD, TTS, and tools actually interacted.

## Evaluation notes

Local evaluation uses deterministic database, action, and communication checks. Strict Tau2 `NL_ASSERTION` scoring was not available because the judge requires a separate compatible provider configuration. The report labels this limitation instead of presenting local reward as strict reward.

The batch also contains retries, a recovered transcript without audio, incomplete metadata-only simulation directories, and mixed train/test provenance. Those artifacts are retained because removing them would make the improvement history look cleaner than the system actually was.

## What I would improve next

- Run multiple seeds for every important task; voice and llm behavior is not deterministic.
- Add a text-only control with the same tools and prompt to separate reasoning failures from speech failures.
- Enable the strict NL judge and report full Tau2 reward.
- Add more return, cancellation, payment, and escalation tasks.
- Add regression tests around spelling retries, ordered variant selection, original-payment refunds, confirmation, termination, and audio-to-turn alignment.
- Measure provider request ids, token usage, time to first token, time to first audio, stt latency, and per-turn cost.
- Test a real acoustic transport and barge-in behavior instead of relying only on the in-process virtual bus.

## Verification

```powershell
python -m ruff check src viewer
```

The project also contains focused tests under `tests/` and a longer technical account in [TECHNICAL_WRITEUP.md](TECHNICAL_WRITEUP.md). The presentation outline is in [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

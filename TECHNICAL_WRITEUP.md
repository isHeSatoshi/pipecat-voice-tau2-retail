# Pipecat + Tau-bench Retail Voice Evals

## Technical write-up

This repository contains a cascaded, real-time voice agent connected to the Tau-bench retail environment. The goal is not only to produce a plausible customer-service conversation. The goal is to make the conversation reproducible, grounded in real tools, observable at the audio/trace level, and honest about where the agent fails.

The current report is based on the retained `v4_calibrated_batch` bundle: 19 recorded attempts across 10 unique task ids, including retries, failures, a recovered transcript, and selected test tasks. The raw artifacts remain available under `data/runs/`; the compact interviewer-facing report is under [`docs/report/`](docs/report/).

## 1. System under test

The tested stack is:

- MiniMax M2.7 for the customer-service agent and the simulated customer;
- NVIDIA Parakeet TDT 0.6B v3 for streaming speech-to-text;
- Chatterbox for text-to-speech;
- Silero VAD for speech boundaries and turn detection;
- Pipecat 1.11 for the real-time voice pipeline;
- Tau-bench retail for tasks, environment state, tools, policy, and local evaluation.

The architecture is intentionally cascaded:

```text
customer LLM
    ↓
customer TTS
    ↓
virtual PCM bus
    ↓
Parakeet STT + Silero VAD
    ↓
customer-service agent LLM
    ↓
grounded Tau-bench tools
    ↓
Tau-bench database state
    ↓
agent TTS
    ↓
next customer turn
```

The two Pipecat pipelines run in one process. The virtual transport has a direction for each side, captures separate agent and user audio, and records lifecycle, STT, VAD, LLM, tool, TTS, and error events in `voice_trace.jsonl`.

## 2. What the agent is allowed to do

The agent must not invent retail state. It authenticates the customer, reads the order, reads product variants, follows the customer’s ordered preferences, states the exact proposed write, waits for an immediate confirmation, executes one write, and reports the actual result.

The `v4` prompt is organized as a conversation state machine:

1. authenticate until a user id is returned;
2. locate the correct order;
3. read current items, product variants, and payment methods;
4. resolve preferences such as low brightness and `battery → USB → AC`;
5. discard stale proposals when the customer changes the request;
6. state the exact item ids, price difference, and payment method;
7. ask for an explicit yes;
8. execute exactly one write;
9. claim completion only after the tool succeeds.

The runtime adds checks that a prompt cannot provide safely. Tool execution is serialized, exact repeated calls are blocked, malformed arguments are rejected before mutation, and a retail write requires an immediately preceding affirmative response in the live conversation.

## 3. The debugging path

The baseline failure was a systems failure, not only a prompt failure.

The first real-stack runs exposed several independent problems:

- Silero VAD was not initialized consistently with the selected sample rate, so speech boundaries were unreliable.
- Tool schema generation read the wrong Tau-bench metadata and exposed empty argument definitions to the model.
- Repeated calls, including writes, were allowed to reach the environment.
- Sentence-level TTS gaps were interpreted as completed customer turns.
- The user simulator did not initially follow Tau-bench’s progressive-disclosure voice guidelines.
- Timeouts were mislabeled, tool errors lost their error state, and traces did not contain enough frame-level evidence.
- Audio was not initially available for transcript review.

The fixes were layered in the same order as the failures:

### Runtime and transport

- Initialize Silero at the pipeline sample rate and release analyzer resources during cleanup.
- Buffer user speech until a VAD stop or a safe silence gap.
- Serialize the virtual transport and prevent input audio from recirculating through the output bus.
- Save exact per-turn audio for new runs instead of estimating clip boundaries from wall-clock timing.
- Preserve legacy audio artifacts and disable unverifiable legacy per-message links rather than guessing.

### Tool bridge

- Build schemas from Tau-bench’s actual `Tool.params` model.
- Preserve required fields, enums, descriptions, and list item types.
- Reject missing and unknown arguments before the environment can mutate.
- Block exact `(tool, normalized arguments)` repeats.
- Preserve structured tool errors in Tau-bench `ToolMessage.error`.

### Agent and user policies

- Ask for a first name and last name one letter at a time after an unclear lookup.
- Ask for the zip code digit by digit.
- Treat ordered preferences as a strict priority list.
- Ask for confirmation only after exact write details are grounded in tool results.
- Ignore non-substantive hold phrases such as “checking, just a sec”.
- Follow the customer’s latest request instead of executing an obsolete proposal.

No task-specific customer identity or product id is hardcoded into the agent.

## 4. Evaluation design

The harness runs deterministic offline checks against saved trajectories. These checks are not a replacement for the full Tau-bench NL judge; they are fast behavioral signals that make debugging concrete.

### `auth_loop`

Fails when the same authentication call repeats with identical arguments, or when the agent asks for authentication details repeatedly after the customer has already supplied them.

### `tool_argument_integrity`

Fails on missing required arguments, unknown arguments, empty payloads, recorded tool errors, missing-argument errors, and not-found responses.

### `write_protocol`

Fails when the agent emits a multi-call tool batch, attempts a write without an immediately preceding explicit confirmation, or ends without attempting the required action.

These checks are useful because a task can have a high local reward while still violating a policy. The calibrated report contains a clear example: task 12 reached local reward `1.0`, but attempted a PayPal refund where the original payment method was required. The write-protocol check correctly marked that run as a failure.

## 5. Current calibrated results

The authoritative report is [`docs/report/analysis.md`](docs/report/analysis.md). It is derived from `data/runs/v4_calibrated_batch/summary.json` and the retained simulation artifacts.

| Metric | Result |
|---|---:|
| Recorded attempts | 19 |
| Unique task ids | 10 |
| Local reward `1.0` | 9/19, 47.4% |
| Database match | 9/19, 47.4% |
| Timeouts | 6/19, 31.6% |
| All three behavior checks passed | 3/19 |
| Strict Tau2 reward available | 0/19 |

The bundle deliberately contains more than one kind of evidence:

- task 0 is a clean local success with all checks passing;
- task 1 has a successful local state but authentication-check warnings;
- task 2 is a recovered transcript with no audio;
- task 3 and task 6 include both failed and successful attempts;
- task 5 is a clear failure across four recorded attempts;
- task 7 is a successful exchange example with separate agent and customer audio;
- task 9 is a qualified local success with a recorded authentication error;
- task 12 exposes a payment-policy failure despite a local score of `1.0`.

The report separates representative results from the full attempt table. It does not hide retries, incomplete metadata-only simulation directories, or the mixed train/test provenance of the bundle.

Charts and tables:

- [`docs/report/charts/task_outcomes.png`](docs/report/charts/task_outcomes.png)
- [`docs/report/charts/attempt_outcomes.png`](docs/report/charts/attempt_outcomes.png)
- [`docs/report/charts/check_pass_rates.png`](docs/report/charts/check_pass_rates.png)
- [`docs/report/charts/attempt_durations.png`](docs/report/charts/attempt_durations.png)
- [`docs/report/charts/representative_check_matrix.png`](docs/report/charts/representative_check_matrix.png)
- [`docs/report/task_summary.csv`](docs/report/task_summary.csv)
- [`docs/report/attempts.csv`](docs/report/attempts.csv)
- [`docs/report/checks.csv`](docs/report/checks.csv)
- [`docs/report/provenance.json`](docs/report/provenance.json)

A portable task 7 example is included as separate files:

- [`conversation_agent_task7.txt`](docs/report/conversation_agent_task7.txt)
- [`conversation_user_task7.txt`](docs/report/conversation_user_task7.txt)
- [`conversation_agent_task7.wav`](docs/report/audio/conversation_agent_task7.wav)
- [`conversation_user_task7.wav`](docs/report/audio/conversation_user_task7.wav)

## 6. Why these design choices

### Cascaded speech

A native audio model may reduce latency, but a cascaded system exposes the exact transcript, tool call, tool result, and generated response. That visibility is more valuable for this project because the goal is to debug policy and pipeline behavior.

### Virtual transport

The in-process transport makes the experiment repeatable and easy to inspect. It does not model acoustic echo cancellation or real network conditions, so its results should not be presented as production telephony measurements.

### One model for both sides

Using MiniMax for the agent and user simulator keeps the setup controlled and affordable. It also means the two sides can share failure modes. A separate user model or scripted user would be a useful ablation.

### Local evaluation

Database, action, and communication checks are deterministic, inexpensive, and useful for iteration. The strict Tau2 NL assertion judge was not configured in this environment, so local reward is explicitly labeled as local reward rather than presented as strict Tau2 reward.

## 7. Reproducing the work

The project expects Python 3.12 and configured model credentials.

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,viewer]"
copy .env.example .env
```

Set `MINIMAX_API_KEY` in `.env`.

Run the offline smoke test:

```powershell
python -m pipecat_voice.cli run --domain retail --task 0 `
  --stt dummy --tts dummy --agent-llm dummy --user-llm dummy `
  --max-seconds 8 --out data/runs/dummy_smoke
```

Run a real task:

```powershell
python -m pipecat_voice.cli run --domain retail --task 7 `
  --num-trials 1 --max-seconds 480 --prompt-variant v4 `
  --agent-llm minimax --user-llm minimax `
  --stt parakeet --tts chatterbox --seed 42 `
  --out data/runs/v4_new_task --check all
```

Analyze saved artifacts without loading models:

```powershell
python -m pipecat_voice.cli analyze --run data/runs/v4_calibrated_batch
```

Open the debugging dashboard:

```powershell
streamlit run viewer/app.py
```

The dashboard provides the transcript, tool calls, reward breakdown, trace timeline, separate agent/user audio, run comparison, and retained failed attempts.

## 8. Next experiments

1. Run paired seeds for every important task because both speech and LLM behavior are noisy.
2. Add a text-only control with identical tools and policy to separate reasoning failures from speech failures.
3. Enable the strict NL judge and report full Tau2 reward.
4. Add more return, cancellation, payment, and human-escalation tasks.
5. Add regression tests for spelling recovery, zip-code recovery, ordered variant selection, original-payment refunds, confirmation, termination, and audio alignment.
6. Record provider request ids, token usage, time to first token, time to first audio, STT latency, and per-turn cost.
7. Test a real acoustic transport and barge-in behavior.

## 9. Public artifact map

- [`README.md`](README.md): human-facing project overview, setup, results, and demo path.
- [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md): short presentation walkthrough.
- [`docs/report/analysis.md`](docs/report/analysis.md): calibrated batch interpretation.
- [`docs/report/`](docs/report/): charts, tables, provenance, and portable conversation examples.
- [`data/runs/`](data/runs/): retained trajectories, traces, audio, and run summaries.
- [`viewer/app.py`](viewer/app.py): Streamlit dashboard.
- [`src/pipecat_voice/`](src/pipecat_voice/): pipeline, transport, Tau-bench bridge, policies, and evaluation code.
- [`tests/`](tests/): focused behavioral and smoke tests.

The project is intentionally evidence-heavy. An interviewer should be able to start with the README, open the charts, inspect the trace timeline, listen to the separate voices, and then follow the code path from audio frame to tool result to database check.

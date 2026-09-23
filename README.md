# Pipecat Voice + Tau-bench Retail Evals

A cascaded Pipecat voice-agent evaluation harness connected to Tau-bench's retail environment, tasks, tools, policy, and evaluator.

## Stack

- MiniMax M2.7 for the agent and user simulator
- NVIDIA Parakeet TDT 0.6B v3 for STT
- Chatterbox for TTS
- Pipecat 1.11
- Tau-bench retail tasks and environment

## Setup

```powershell
cd D:\Project\infer_task\pipecat_voice
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -e ../tau2-bench
uv pip install -e ".[dev,viewer]"
copy .env.example .env
```

Set `MINIMAX_API_KEY` or `ANTHROPIC_API_KEY` in `.env`. The default MiniMax base URL is `https://api.minimax.io/anthropic`.

## Run

```powershell
# Offline dummy smoke
python -m pipecat_voice.cli run --domain retail --task 0 `
  --stt dummy --tts dummy --agent-llm dummy --user-llm dummy

# Focused real-stack run
python -m pipecat_voice.cli run --domain retail --task 1 --num-trials 1 `
  --max-seconds 180 --prompt-variant v4 `
  --agent-llm minimax --user-llm minimax `
  --stt parakeet --tts chatterbox --seed 42 `
  --out data/runs/v4_guard --write-summary --check all

# Analyze saved trajectories without API or GPU use
python -m pipecat_voice.cli analyze --run data/runs/baseline_m27
```

## Viewer

```powershell
streamlit run viewer/app.py
```

The viewer shows transcript, reward, event timeline, tool calls, run comparison, auto-refresh, and agent/user audio playback.

## Architecture

```text
user LLM → user TTS → virtual PCM bus → agent STT → agent LLM
                                                    ↓
                                            Tau2 environment
                                                    ↓
                                            agent TTS → PCM bus
```

The Tau2 environment executes tools while Pipecat streams the conversation. At the end, the agent context is converted into a Tau2 `SimulationRun` and scored locally with ENV, ACTION, and COMMUNICATE evaluators.

The virtual transport is a local, cascaded voice loop. It initializes Silero VAD, records PCM audio, and emits real STT, LLM, tool, TTS, error, and lifecycle events.

## Outputs

Each simulation writes:

```text
data/runs/<run>/task_<id>/sim_<uuid>/
├── trajectory.json
├── voice_trace.jsonl
├── agent_audio.wav
├── user_audio.wav
└── summary data at run root
```

`trajectory.json` contains the canonical Tau2 messages, tool calls, reward, seed, termination reason, and evaluation note. `voice_trace.jsonl` contains conversation-relative events and prompt/model provenance.

## Three failure-mode evals

| Check | Failure |
|---|---|
| `auth_loop` | Identical authentication calls or repeated requests for already supplied details |
| `tool_argument_integrity` | Empty, missing, unknown, or failed tool arguments |
| `write_protocol` | Multi-call batches, writes without immediate confirmation, or missing required actions |

Checks are pure offline predicates and run automatically at the end of every simulation.

## Prompt variants

- `baseline`: Tau2-compatible baseline instructions plus retail policy
- `v1`: authentication and turn discipline
- `v2`: explicit transition after authentication
- `v3`: identifier and tool-error recovery
- `v4`: voice state machine, grounded IDs, single-call execution, confirmation, and completion protocol

Use `--prompt-variant v4` for the current improved condition.

## Current results

The five saved MiniMax M2.7 baseline runs produced:

- 0/5 strict successes
- 100% authentication-loop failures
- 100% write-protocol failures
- 80% tool-argument-integrity failures
- 100% fragmented user turns
- 105.2 mean messages per run
- 300.9 seconds mean duration

The focused v4 + runtime-guard task 1 conversation authenticated the user, made six grounded tool calls, obtained explicit confirmation, successfully exchanged thermostat item `4983901480` for `7747408585`, and passed all three behavior checks. The recorded run timed out only because the simulator said “That’s all I needed” without emitting `###STOP###`; natural completion detection now ends that phrase as `user_stop`.

See `TECHNICAL_WRITEUP.md` for the full analysis, trade-offs, evidence, and next experiment. See `DEMO_SCRIPT.md` for the under-five-minute presentation outline.

## Evaluation limitations

- NL assertions are not scored. When a task requires them, strict reward is forced to zero and the partial ENV/ACTION/COMMUNICATE product is stored separately.
- Existing baseline cells are single trials and should not be treated as statistically significant.
- The virtual transport is suitable for closed-loop behavior testing but does not model acoustic echo cancellation.

## Verification

```powershell
python -m pytest -q
python -m ruff check src viewer tests/test_*.py
python -m ruff format --check src viewer tests/test_*.py
```

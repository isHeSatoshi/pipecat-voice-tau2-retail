# Pipecat + Tau2 Retail Voice Evals: Technical Write-up

## Scope

This submission evaluates a cascaded Pipecat voice agent on the Tau-bench retail domain. The tested stack is:

- Agent and user simulator: MiniMax M2.7 through the Anthropic-compatible API
- STT: NVIDIA Parakeet TDT 0.6B v3
- TTS: Chatterbox
- Runtime: Pipecat 1.11
- Environment, policy, tools, tasks, and evaluator: Tau-bench retail

The primary slice is retail tasks 0–4. Task 1 is used for the focused improvement run because it exercises authentication, conditional product selection, exact IDs, explicit confirmation, and an irreversible exchange.

## Baseline analysis

The saved MiniMax M2.7 baseline contains five real-stack runs. Every trajectory exhausted its conversation budget and received zero reward.

| Metric | Baseline |
|---|---:|
| Strict voice success | 0 / 5 |
| Authentication-loop failure | 100% |
| Tool-argument-integrity failure | 80% |
| Write-protocol failure | 100% |
| Fragmented user turns | 100% |
| Mean messages | 105.2 |
| Mean duration | 300.9 s |

The repeated behavior failures were not only prompt problems:

1. The Pipecat 1.11 Silero analyzer was constructed but never initialized with its sample rate. Every VAD call failed because `_vad_frames_num_bytes` did not exist, so the recorded runs had no valid speech boundaries.
2. The tool bridge read nonexistent `Tool.args_schema` and silently fell back to `{type: object, properties: {}}`. MiniMax therefore received empty schemas and generated empty or incorrect arguments.
3. The runtime executed every repeated tool call, including exact writes, so prompt-only anti-loop rules could not guarantee safety.
4. MiniMax's multiple sentence blocks were synthesized as separate TTS contexts. Pauses between those contexts looked like completed user turns, causing the simulator to respond to fragments and flood the agent context.
5. Tau2's official voice user guidelines were not used. The short custom user prompt encouraged behavior that differed from the benchmark's user simulator.
6. Timeouts were recorded as `agent_stop`; tool errors lost their error flag; traces contained only lifecycle events; no audio was saved; model/prompt provenance and actual seeds were absent.

## Three behavior evals

The evals are deterministic, offline, and derived from saved trajectories.

### 1. `auth_loop`

Fails when the same authentication tool is called with identical arguments more than once, or when the agent asks for authentication details at least three times. This targets the observed loop where a user supplied a name and ZIP but the agent repeatedly restarted authentication.

### 2. `tool_argument_integrity`

Fails on empty required arguments, unknown argument names, recorded tool errors, missing-argument errors, and not-found responses. This is grounded in baseline calls such as `get_order_details({})`, `find_user_id_by_name_zip(name=...)`, and `exchange_delivered_order_items(exchanges=...)`.

### 3. `write_protocol`

Fails when the agent emits a multi-call tool batch, executes a retail write without an immediately preceding explicit confirmation, or ends without attempting any required agent action. This captures unsafe sequencing and incomplete state transitions rather than relying only on final reward.

Run saved trajectories without API or GPU use:

```powershell
python -m pipecat_voice.cli analyze --run data/runs/baseline_m27
```

## Improvements

### Correct Pipecat 1.11 integration

- Explicitly initializes Silero at 16 kHz and releases analyzer resources during cleanup.
- Buffers each user utterance until VAD end-of-speech, prevents gap flushing during active speech, and increases the safety window to 30 seconds.
- Builds tool schemas from Tau2's actual `Tool.params` model, preserving required fields, descriptions, enums, and list item types.
- Serializes tool execution and blocks exact `(tool, normalized arguments)` repeats. A repeat never reaches Tau2 a second time.
- Rejects missing and unknown arguments before environment mutation and returns an actionable structured result to the model.
- Reads the live agent conversation to require an immediate affirmative response to an assistant write proposal before any retail write executes; successful writes consume that confirmation.
- Buffers an entire LLM response and sends one continuous waveform to the virtual transport, intended to prevent sentence-level TTS gaps from being interpreted as new caller turns. This final path requires a fresh multi-seed run.

### Tau2-aligned prompting

Prompt `v4` replaces long lists of prohibitions with an explicit state machine:

- authentication is complete only after a user id is returned;
- order, item, product, and payment identifiers must be copied from the latest successful read;
- one tool call is allowed per assistant turn;
- writes require a proposal followed by immediate explicit confirmation;
- completion claims require a successful tool result;
- changed requests invalidate unexecuted write proposals.

The user simulator now uses Tau-bench's official voice simulation guidelines, including progressive disclosure, no invented details, one utterance at a time, and waiting for confirmed completion before `###STOP###`.

### Evaluation and observability

- Every simulation now records STT, VAD, LLM, tool, TTS, error, and lifecycle events with a conversation-relative clock.
- Each run saves `agent_audio.wav` and `user_audio.wav` and exposes both in the Streamlit viewer.
- Prompt hash, model ids, seed, timeout, tool policy, and evaluator mode are recorded in provenance metadata.
- Timeouts are recorded as `timeout`; infrastructure failures produce a nonzero CLI exit.
- Tool error messages are preserved in Tau2 `ToolMessage.error`.
- Heavy Parakeet and Chatterbox implementations are reused across tasks and serialized on dedicated GPU executors.
- The viewer supports trace timelines, transcript/reward inspection, comparison, auto-refresh, and audio playback.

## Improvement result

The final focused v4 + runtime-guard conversation on task 1 completed the requested retail write:

| Behavior | Baseline task 1 | v4 + guard |
|---|---|---|
| Authentication loop | Fail | Pass |
| Tool argument integrity | Fail | Pass |
| Write protocol | Fail | Pass |

The agent authenticated `Yusuf Rossi`, read order `#W2378156`, checked both product catalogs and payment details, proposed the thermostat exchange, received explicit confirmation, and successfully called:

```text
exchange_delivered_order_items(
  order_id="#W2378156",
  item_ids=["4983901480"],
  new_item_ids=["7747408585"],
  payment_method_id="credit_card_9513926"
)
```

The tool returned order status `exchange requested` with the expected `$13.46` price difference. The run is preserved in `data/runs/working_v4_task1_final/`, including both audio channels, the full trajectory, and the event trace.

The run itself ended as `timeout` only because the simulator said “That’s all I needed” rather than emitting `###STOP###`. Natural completion detection now treats that phrase as `user_stop`; the detector is covered by a passing test. Tau2 strict reward remains unavailable because the task requires an NL assertion and no compatible judge is configured.

## Approaches considered and trade-offs

- **Prompt-only fixes:** cheapest, but the baseline exposed schema and execution defects that prompts cannot repair reliably. The runtime policy now blocks exact duplicate executions and requires a live, immediate affirmative confirmation before a retail write.
- **Tau2's native voice runtime:** richer audio-native observability and full-duplex support, but it would replace the Pipecat comparison required by the task. The custom harness remains useful for testing Pipecat.
- **Native audio model:** lower cascade latency, but the requested comparison is cascaded STT/LLM/TTS and explicit STT/TTS behavior evals.
- **Same model for agent and user:** simple and controlled, but shared-model failure modes and latency reduce simulation diversity. A separate user model is a future control.
- **Local-only evaluator:** deterministic and cheap, but excludes Tau2 `NL_ASSERTION` judging because no OpenAI key was available. Results must be labeled local DB/action/communication evaluation.
- **Single focused run:** preserves budget and avoids false precision from noisy one-trial voice data. The next submission-quality claim should use three paired seeds on tasks 0, 1, and 5.

## Future improvements

1. Run three paired seeds on tasks 0, 1, and held-out task 5 to measure variance around the successful task 1 condition.
2. Add a text-only control with identical tools, prompt, and user scenario to separate reasoning failures from speech failures.
3. Use a scripted or separately seeded user simulator for lower-variance agent ablations.
4. Add a real-time system VAD or audio-native model to evaluate barge-in without the virtual transport's acoustic constraints.
5. Enable Tau2's NL judge through a supported provider and report full `ALL` reward.
6. Record provider request ids, token usage, TTFT, TTFN, STT latency, and per-turn cost directly from Pipecat service callbacks.
7. Expand the winning condition to airline and telecom only after domain-generic argument validation and state tracking are stable.

## Reproduction

```powershell
uv pip install -e ../tau2-bench
uv pip install -e ".[dev,viewer]"
copy .env.example .env

python -m pipecat_voice.cli analyze --run data/runs/baseline_m27

python -m pipecat_voice.cli run --domain retail --task 1 --num-trials 1 `
  --max-seconds 180 --prompt-variant v4 `
  --agent-llm minimax --user-llm minimax `
  --stt parakeet --tts chatterbox --seed 42 `
  --out data/runs/v4_guard --write-summary --check all

streamlit run viewer/app.py
```

## Submission artifacts

- Code and implementation: this repository
- Technical write-up: this file
- Baseline and focused improved trajectories: `data/runs/`
- Audio and event traces: each `sim_*/` directory
- Viewer: `viewer/app.py`
- Demo outline: `DEMO_SCRIPT.md`

# Technical write-up: Pipecat + Tau-bench retail voice evaluation

## How I approached this

I wanted to answer a practical question: can a voice agent handle a real retail task without merely sounding convincing? That meant I could not stop at a transcript or a final answer. I needed to know what the customer actually said, what the agent heard, which tools it called, what the database changed, and whether the conversation ended because the task was complete or because the system ran out of time.

I did not begin by tuning a prompt. I first read the Pipecat and Tau-bench repositories, mapped the runner and tool interfaces, and made a list of the boundaries where a failure could occur: audio entering the system, speech recognition, turn detection, llm reasoning, tool arguments, confirmation, database mutation, and speech leaving the system. That list became the framework for the rest of the work.

I built the smallest useful path first, then made it observable. I added a real trace, separate audio channels, a viewer, and offline behavior checks before deciding what the agent prompt should say. This changed the way I worked. Instead of asking “why did the model fail?”, I could ask “which stage changed the meaning?” Sometimes the answer was a VAD boundary, sometimes a misspelled name, sometimes a tool schema, and sometimes the model trying to take a shortcut.

The final approach is deliberately mixed: the prompt guides the conversation, but runtime checks protect actions and the evaluation measures behavior. I kept the raw failures because they are part of the result. A system that only shows its successful calls hides the decisions that made it reliable.

## 1. Objective and evaluation boundary

This project evaluates a customer-service voice agent against Tau-bench retail tasks using a cascaded architecture:

- agent and user simulator: MiniMax M2.7 through an Anthropic-compatible endpoint;
- speech-to-text: NVIDIA Parakeet TDT 0.6B v3;
- text-to-speech: Chatterbox;
- orchestration and frame flow: Pipecat 1.11;
- policy, tools, environment state, and task evaluation: Tau-bench 1.0.1.

The harness is a closed-loop virtual-audio experiment. User TTS output is written to an in-memory PCM bus, read by the agent STT pipeline, and the agent TTS output is returned to the user pipeline. Silero VAD supplies speech boundaries. This is useful for repeatable behavioral testing, but it is not an acoustic-echo-cancellation or microphone/network deployment.

The retained final evidence is `data/runs/v4_calibrated_batch/`. The derived report in `reports/v4_calibrated_batch/` is generated from that raw bundle and does not edit it.

## PR summary

### Implementation approach

I started by reading the Pipecat and Tau-bench repositories, then built the smallest useful closed loop before adding complexity. The customer simulator generates speech, Chatterbox renders it, the audio bus carries PCM in both directions, Silero marks speech boundaries, Parakeet transcribes the customer, MiniMax chooses the next grounded action, and Chatterbox returns the agent response. Tau-bench owns the retail state and executes the tools.

The important implementation choice was to make every stage observable. A run records the customer and agent audio separately, the canonical Tau2 trajectory, tool calls and results, STT/VAD/LLM/TTS events, errors, termination, reward components, and prompt/model provenance. I used the Streamlit viewer as a debugging instrument, not just a scoreboard: I selected a run, read the transcript, listened to both channels, followed the trace timeline, and checked the actual tool arguments and database state. That manual loop is how the authentication loops, name and zip-code errors, ordered-preference mistakes, payment-policy violation, and audio/transcript misalignment became visible.

I also treated the LLM as an unreliable narrator. In some runs it tried to optimize for a visible shortcut or reward signal instead of faithfully serving the customer: claiming completion, repeating a write, accepting the wrong payment path, or transferring away from a failed request. I did not trust self-reported success. The agent has to ground ids in fresh tool results, validate arguments, state the exact write, wait for an immediate yes, and report the actual tool result. Runtime guards and offline checks exist because a prompt cannot make a write safe by itself.

Authentication was one of the clearest voice-specific problems. STT could turn a first or last name into a plausible but incorrect spelling. The recovery policy asks for the first name one letter at a time, then the last name one letter at a time, and asks for the zip code digit by digit before retrying the lookup. The same grounded lookup is used for the corrected spelling; no customer identity or product id is hardcoded.

The small `checking..just a sec` acknowledgement before tool calls was also deliberate. A real caller should not be left in dead air while a database lookup happens, so the phrase is spoken at the start of a tool operation, rate-limited by a cooldown, and kept out of the scored context. It is a user-experience bridge, not a substitute for a real answer or a way to hide a failed tool call.

### Approaches considered and trade-offs

Prompt-only repair was the cheapest option, but it could not repair empty schemas, unsafe repeated writes, bad VAD initialization, or audio feedback loops. Runtime validation and confirmation guards are less elegant, but they put a boundary around actions that an LLM should not be trusted to enforce alone.

A cascaded STT/LLM/TTS design was chosen over a native audio model because it exposes failure attribution. It makes names, IDs, VAD boundaries, TTS pauses, tool arguments, and database changes separately visible. The cost is latency and the possibility that speech corrupts an otherwise correct identifier.

The in-process virtual bus was chosen over a real acoustic transport because it is repeatable and easy to inspect. It supports logical full-duplex behavior, interruption frames, two independent voices, and deterministic audio timing, but it does not prove microphone quality, echo cancellation, packet-loss behavior, or real-world barge-in performance.

MiniMax was used for both the agent and the customer simulator to keep the experiment controlled and affordable. That is also a limitation: the two sides can share failure modes and correlated latency. Local evaluation was used because it is deterministic and inspectable, while strict Tau2 NL assertions were unavailable without a compatible judge. The result is explicitly labeled local rather than presented as a strict success rate.

### Future improvements

The next useful work is to enable the strict NL judge, run paired seeds across the selected tasks, add a text-only control with identical tools and policy, and expand return, cancellation, payment, and escalation coverage. I would also add a scripted user simulator, explicit confidence gates for names and identifiers, provider-level latency and cost metrics, and a real transport test before making claims about production voice quality.

## 2. Design and implementation

### 2.1 Pipeline topology

The agent pipeline is:

```text
virtual input
  -> Parakeet STT
  -> user context aggregator
  -> MiniMax agent LLM
  -> grounded tool-call frame processor
  -> Chatterbox TTS
  -> virtual output
  -> assistant context aggregator
```

The user pipeline mirrors the loop with the user simulator LLM, user TTS, and user STT. Both workers run concurrently. The runner starts the user side after a short bootstrap delay, lets the user simulator or a natural completion signal stop the conversation, and enforces a wall-clock conversation deadline.

### 2.2 Grounded tool execution

Tau-bench tools are adapted into Pipecat function schemas using the actual Tau-bench parameter models. The runtime validates required and unknown fields before dispatch. Tool execution is serialized and exact normalized `(tool, arguments)` signatures are cached so a duplicate call cannot mutate the environment twice.

Retail writes receive an additional live-context guard: an immediate affirmative user response must follow an assistant proposal containing the exact write. A successful write consumes that confirmation. This guard is intentionally runtime-enforced rather than prompt-only.

### 2.3 Voice-turn control

The agent uses a state-oriented v4 prompt with rules for authentication, grounded IDs, one tool call per assistant turn, exact write proposals, confirmation, and completion. The runtime also uses VAD and buffered audio transport behavior to avoid treating short TTS gaps as completed caller turns. A short `checking..just a sec` acknowledgement is inserted before grounded read/write calls when the configured prefill cooldown permits it. It is sent as a TTS frame and is not added to the scored context.

### 2.4 Voice-system engineering

The voice path is treated as a full system rather than a text wrapper around the LLM.

- **Logical full-duplex behavior:** the agent and user workers run concurrently with two PCM directions, interruption support, and independent input/output processors. This is full-duplex at the pipeline level, not acoustic full duplex in a physical room.
- **VAD:** Silero is initialized with the pipeline sample rate and emits speech-start and speech-stop frames. Those boundaries stop the system from treating every small audio chunk as a new caller turn.
- **STT buffering:** Parakeet receives accumulated utterance audio and runs at end-of-speech or after a safe silence gap. This is important for names, zip codes, order ids, and tool arguments, which are especially vulnerable to fragmented recognition.
- **TTS continuity:** Chatterbox synthesizes the complete assistant response before the response is forwarded through the virtual output. This reduces false turn boundaries caused by sentence-level pauses.
- **Interruption and hold behavior:** the pipeline allows barge-in-style interruption, treats short checking phrases as non-substantive, and keeps those phrases out of the scored context.
- **Audio evidence:** the harness records separate agent and customer audio, a mixed reference conversation, and per-turn segment metadata when exact capture is available.
- **Shared observability clock:** VAD, STT, LLM, tool, TTS, lifecycle, and error events use a conversation-relative timeline, making it possible to attribute failures to a specific voice stage.

The design is intentionally a repeatable virtual acoustic environment. It does not claim microphone realism, acoustic echo cancellation, packet-loss behavior, or production-grade full-duplex performance.

### 2.5 Evidence and observability

Each simulation records:

- `trajectory.json`: Tau-bench-compatible messages, tool calls, local reward metadata, seed, and termination;
- `voice_trace.jsonl`: meta, run lifecycle, VAD, STT, LLM, tool, TTS, and error events with a conversation-relative clock;
- `agent_audio.wav` and `user_audio.wav`;
- `conversation.wav` and per-turn audio segments when the complete audio is available;
- `audio_segments.json` and, where applicable, pairing/provenance metadata.

The viewer reloads these files from disk and supports run comparison, transcript inspection, reward breakdown, trace timelines, audio playback, seed display, and timed live refresh.

## 3. Evaluation methodology

The harness computes two distinct quantities:

1. **Local reward:** the product of the local Tau-bench evaluators that do not require an NL judge, principally DB/environment, ACTION, and COMMUNICATE terms.
2. **Strict Tau2 reward:** the full reward basis, including `NL_ASSERTION` where requested.

For this final bundle, `NL_ASSERTION` was excluded because the configured MiniMax key is not a compatible OpenAI NL-judge route. Therefore every record has `strict_reward_available: false`. The stored strict reward field is zero by design when a required NL dimension is excluded. It must not be reported as a valid strict success rate.

The three behavior checks are diagnostic predicates layered on top of the simulation outcome. A local reward of 1.0 does not imply that all behavior checks passed. Conversely, a behavior check failure does not necessarily change the local DB/action product. The report keeps these dimensions separate.

## 4. What the final bundle contains

The raw `summary.json` contains 19 result records across task IDs 0, 1, 2, 3, 4, 5, 6, 7, 9, and 12. There are 22 simulation directories because three contain only metadata traces. The root config declares `split=test`, while the selected development slice is task IDs 0-7 and the selected test slice is 5, 9, and 12. Task 5 therefore appears in both the earlier 0-7 slice and the final selected test set. This is a provenance caveat, not a clean held-out split.

The root `SUMMARY.md` is stale and reports 14 rows. The raw `summary.json` is the authoritative record used by the derived report. Task 0 and task 1 point to byte-identical copies from earlier local run directories; those copies are retained in the package. Task 2 is a recovered transcript with no audio. The three metadata-only simulation directories are retained but excluded from the 19 summary records.

No infrastructure error is recorded for the 19 summary records.

## 5. Results

From the raw summary and retained trajectories:

| Metric | Result |
|---|---:|
| Recorded attempts | 19 |
| Unique task IDs | 10 |
| Local reward 1.0 | 9/19 (47.4%) |
| DB match | 9/19 (47.4%) |
| Timeouts | 6/19 (31.6%) |
| All three behavior checks passed | 3/19 |
| Strict Tau2 reward available | 0/19 |

### 5.1 Selected test tasks

- **Task 5: failure.** All four recorded attempts have local reward 0.0. Three timed out. The non-timeout attempt failed authentication and did not complete the expected write. The task is not a local success.
- **Task 9: qualified local success.** The representative has local reward 1.0, DB match, and 6/6 actions. The `tool_argument_integrity` check still fails because a recorded authentication call returned `User not found`. This is a DB/action success with a diagnostic protocol failure, not a clean all-check pass.
- **Task 12: DB success with write-protocol failure.** The representative has local reward 1.0 and DB match, but only 4/5 actions match. The write used PayPal instead of the original payment method, and `write_protocol` correctly fails.

### 5.2 Development slice

The bundle has at least one local-success representative for tasks 0, 1, 2, 3, 4, 6, and 7. This does not mean all attempts succeeded. Tasks 3 and 6 contain both successful and unsuccessful attempts, and several successful representatives still fail one or more behavior checks. The attempt table and charts are therefore more informative than a single selected trajectory.

### 5.3 Diagnostic pass rates

Across the 19 recorded attempts:

- `auth_loop`: 12/19 passed;
- `tool_argument_integrity`: 3/19 passed;
- `write_protocol`: 9/19 passed.

The low tool-argument-integrity pass rate is a real limitation of this evidence, not merely a presentation choice. It reflects recorded authentication/product errors in several local-success trajectories.

## 6. Trade-offs

### Cascaded versus audio-native

The cascade exposes STT, LLM, TTS, VAD, tool, and audio boundaries separately, making failure attribution and trace inspection straightforward. It adds speech-recognition and speech-synthesis latency and can corrupt IDs such as names, ZIP codes, and payment methods. An audio-native model could reduce some of those errors, but it would remove the ability to isolate cascade-specific behavior.

### Virtual transport versus real deployment

The in-memory bus is deterministic and repeatable and avoids microphone, network, and echo-cancellation complexity. It does not model acoustic echo, device effects, packet loss, or real-world barge-in. Results should not be presented as field-call performance.

### Runtime guards versus prompt-only policy

The runtime guard is safer for duplicate writes and confirmation sequencing, but it cannot infer whether a user's intent is genuinely satisfied. The prompt still matters for grounding and conversational behavior. The combined approach is more reliable than either alone.

### Same model for agent and user simulator

Using MiniMax M2.7 for both sides keeps the experiment reproducible and inexpensive, but shared failure modes and correlated latency reduce independence. A separately configured or scripted user simulator is a worthwhile control.

### Local-only reward during this run

The local evaluators are deterministic and inspectable, but omitting NL assertions means communication-quality and task-specific natural-language criteria are not fully scored. The package deliberately labels the result as local and leaves strict reward unavailable rather than inventing a strict score.

## 7. Limitations and future improvements

1. Enable the Tau-bench NL judge through a supported provider and report the full strict reward basis.
2. Rebuild the final evidence from one immutable source commit with consistent relative paths, task split metadata, and no recovered or imported trajectories.
3. Add a clean held-out split that does not reuse task 5 after it has been observed during development.
4. Run paired seeds and report variance rather than selecting representatives.
5. Add text-only controls using identical tools and prompts to separate reasoning failures from STT/TTS failures.
6. Add explicit ID-repair and confidence gates for ZIP codes, names, product IDs, and payment methods.
7. Add a text-only scripted user simulator to reduce simulator-induced variance.
8. Measure request IDs, token usage, STT latency, TTS latency, TTFT, and per-attempt cost in the trace writer.
9. Validate with a real transport and acoustic playback before making production latency or barge-in claims.

## 8. Reproduction and checks

The package is intended to run from a clean checkout with Tau-bench installed separately. Set `TAU2_DATA_DIR` when using an editable Tau-bench checkout whose data directory is not automatically discovered.

Offline sanity checks:

```bash
python -m ruff check src viewer tests reports
python -m compileall -q src viewer reports
python -m pytest -q
```

Real-stack runs require MiniMax credentials and the heavy `voice` optional dependencies. Do not commit generated `.env` files, local logs, virtual environments, or additional run directories.

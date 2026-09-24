# Technical write-up: cascaded Pipecat + Tau-bench retail voice evaluation

## 1. Objective and evaluation boundary

This project evaluates a customer-service voice agent against Tau-bench retail tasks using a cascaded architecture:

- agent and user simulator: MiniMax M2.7 through an Anthropic-compatible endpoint;
- speech-to-text: NVIDIA Parakeet TDT 0.6B v3;
- text-to-speech: Chatterbox;
- orchestration and frame flow: Pipecat 1.11;
- policy, tools, environment state, and task evaluation: Tau-bench 1.0.1.

The harness is a closed-loop virtual-audio experiment. User TTS output is written to an in-memory PCM bus, read by the agent STT pipeline, and the agent TTS output is returned to the user pipeline. Silero VAD supplies speech boundaries. This is useful for repeatable behavioral testing, but it is not an acoustic-echo-cancellation or microphone/network deployment.

The retained final evidence is `data/runs/v4_calibrated_batch/`. The derived report in `reports/v4_calibrated_batch/` is generated from that raw bundle and does not edit it.

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

### 2.4 Evidence and observability

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

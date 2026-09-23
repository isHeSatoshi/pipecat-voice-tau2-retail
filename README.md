# pipecat_voice

Full-duplex voice-agent evaluation harness that joins **Pipecat** (the voice runtime) to **tau2-bench** (the retail evaluation side).

## What's in this package

- **Pipecat full-duplex pipeline** with Parakeet STT (`nemo_toolkit[asr]`), Chatterbox TTS (`chatterbox-tts`), and **MiniMax M2.7 High-Speed** as the agent LLM via LiteLLM.
- **Tau2-bench retail** environment, tasks, policy, tools, and evaluator — consumed as a library, never modified.
- **Custom `Tau2ToolBridge`** that intercepts Pipecat's function-call frames and routes them into `tau2.environment.Environment.make_tool_call(...)`, feeding results back into the LLM context.
- **Closed-loop user simulator pipeline** that runs tau2's `UserSimulator` system prompt through a second Pipecat pipeline (STT → user LLM → TTS), with the agent's TTS audio going back through STT into the user LLM.
- **Virtual in-process audio transport** so both pipelines share audio frames in one event loop (no microphone, no websocket — purely local).
- **Pipecat's built-in VAD + interruption** for natural barge-in; tau2's `###STOP###` semantics adapted at the integration boundary.
- **Replaceable STT/TTS/LLM** boundaries behind Python `Protocol`s so each component can be swapped (dummy impls for CI smoke).

## Layout

```
pipecat_voice/
├── pyproject.toml
├── .env.example                       # copy to .env and fill in MiniMax key
├── src/pipecat_voice/
│   ├── config.py                       # env-driven config dataclasses
│   ├── interfaces.py                   # STT/TTS/LLM Protocols
│   ├── services/
│   │   ├── parakeet_stt.py             # Pipecat STTService for Parakeet
│   │   ├── chatterbox_tts.py           # Pipecat TTSService for Chatterbox
│   │   └── litellm_service.py          # MiniMax via LiteLLM (Anthropic-compatible)
│   ├── transport/virtual_transport.py  # in-memory dual-endpoint audio bridge
│   ├── pipelines/
│   │   ├── agent_pipeline.py           # STT → MiniMax LLM → ToolBridge → TTS
│   │   └── user_pipeline.py            # STT → user LLM → TTS
│   ├── tau2/
│   │   ├── environment.py              # build tau2 retail env + tools
│   │   ├── context.py                  # LLMContext from tau2 env (system prompt + tools)
│   │   ├── tool_bridge.py              # FrameProcessor for tool calls
│   │   └── runner.py                   # Tau2EvalRunner (drives both pipelines + scoring)
│   ├── observability/trace_writer.py   # JSONL events + tau2 SimulationRun dump
│   └── cli.py                          # `python -m pipecat_voice.run ...`
└── tests/
    ├── test_smoke.py                   # 1 retail task end-to-end with dummy impls
    └── test_swap_bounds.py             # prove STT/TTS/LLM are replaceable
```

## Setup

```powershell
cd D:\Project\infer_task\pipecat_voice
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -e ../tau2-bench
uv pip install -e ".[dev]"
copy .env.example .env
# edit .env: set MINIMAX_API_KEY (and optionally MINIMAX_API_BASE)
```

## Run

```powershell
# 1 retail task with dummy impls (no GPU / no network) — verifies the loop closes
python -m pipecat_voice.cli run --task 0 --stt dummy --tts dummy --agent-llm dummy --user-llm dummy

# 1 retail task with the real stack (Parakeet + Chatterbox + MiniMax)
python -m pipecat_voice.cli run --task 0 --stt parakeet --tts chatterbox

# batch over the tau2 retail base split
python -m pipecat_voice.cli run --split base --num-trials 1 --max-tasks 10 --out data/runs/baseline
```

Each run writes to `<out_dir>/sim_<id>/`:

| File | Contents |
|---|---|
| `trajectory.json` | tau2 `SimulationRun` (canonical, includes `reward_info`, all messages, tool calls, termination reason) |
| `voice_trace.jsonl` | one event per line: `user_audio_in`, `stt_result`, `llm_call`, `tool_call`, `tool_result`, `agent_audio_out` — each with `turn_idx`, `latency_ms`, `audio_path` |
| `audio/user_turn_<n>.wav` | user-side TTS audio |
| `audio/agent_turn_<n>.wav` | agent-side TTS audio |
| `meta.json` | resolved config snapshot (model, sample rate, seed, task id, trial) |

## Architecture

```
                   ┌─────────────────── user simulator pipeline ───────────────────┐
                   │                                                            │
                   │   user STT → user LLM (LiteLLM/MiniMax) → user TTS → ───┐    │
                   │      ▲                                                  │    │
                   └──────┼──────────────────────────────────────────────────┼────┘
                          │ agent TTS audio                                   │ user TTS audio
                          │ (after STT round-trip)                            │
                          ▼                                                  ▼
                   ┌────────────────────── virtual transport ──────────────────────┐
                   │  in-memory PCM frame queues (one per direction)                │
                   └──────────────────────┬────────────────────┬────────────────────┘
                                          │                    │
                                          ▼                    ▲
                   ┌────────────── agent pipeline ──────────────┴───────────────┐
                   │                                                              │
                   │   STT (Parakeet) → LLMContextAgg → LLM (LiteLLM/MiniMax)   │
                   │                                  │                           │
                   │                                  ▼                           │
                   │                          Tau2ToolBridge  ──► tau2 Environment│
                   │                                  │          (RetailDB, tools)│
                   │                                  ▼                           │
                   │                          AssistantContextAgg → TTS (Chatterbox)
                   └──────────────────────────────────────────────────────────────┘
```

The tau2 environment executes tool calls synchronously while the Pipecat pipeline streams; tau2's evaluator scores the final DB hash and assertions after the conversation ends.

## Notes on full-duplex

This harness uses Pipecat's standard full-duplex primitives — VAD via `SileroVADAnalyzer`, barge-in via `LLMUserResponseAggregator`, transport-driven interruption. It does **not** use tau2's turn-based orchestrator at runtime; tau2's role is the env + tasks + evaluator. `###STOP###` / `###TRANSFER###` / `###OUT-OF-SCOPE###` emitted by the user LLM are detected at the integration boundary (`user_pipeline.py`) and trigger pipeline teardown.

## Why dummy impls

The `dummy` STT/TTS/LLM let CI run an end-to-end smoke test without GPU, without Parakeet/Chatterbox downloads, and without a MiniMax API key. They satisfy the protocols' contracts (return shapes, latencies, transcript coherence) but don't actually run real models. Once dummy works, swap `--stt parakeet --tts chatterbox --agent-llm minimax` and the same loop runs the real stack.

## What this package does NOT do

- Does not modify `tau2-bench` source. tau2 is a library dependency.
- Does not use `tau2.voice.*`, `tau2.voice.audio_native.*`, `DiscreteTimeAudioNativeAgent`, or `VoiceStreamingUserSimulator`.
- Does not implement interruption/backchannel policies beyond Pipecat's built-in defaults in this first cut.
- Does not implement airline/telecom/banking_knowledge domains. Retail only.

## Eval half — failure-mode checks + prompt variants

Beyond the headline reward, the harness exposes a small **failure-mode check** API in `src/pipecat_voice/eval/`. Each check is a pure function over a finished `SimulationRun` + `Task` that flags one specific way the agent gets stuck. Run any subset with the CLI flag:

```powershell
python -m pipecat_voice.cli run --task 0 --stt parakeet --tts chatterbox \
    --agent-llm minimax --user-llm minimax \
    --write-summary --check all
```

`--check` accepts a comma-separated list (`--check auth_loop,no_tool_calls`) or `all`. Results are appended to `SUMMARY.md` as a per-task table. The three checks that ship today:

| Check | Failure mode it catches |
|---|---|
| `auth_loop` | Same `(tool_name, args)` signature appears ≥3 times (or the same `find_user_id_*` call ≥2 times in the first 5 turns). Agent is stuck re-asking the same authentication question. |
| `no_tool_calls` | Agent never invoked a tau2 tool. Pure-text replies usually mean the agent gave up or hallucinated. |
| `premature_stop` | Conversation ended with `user_stop` while the gold trajectory still requires action, with zero agent tool calls. |

Adding a new check is a single function in `src/pipecat_voice/eval/checks.py` plus one line in `pipecat_voice/eval/__init__.py`.

### Prompt variants

System-prompt overrides live in `src/pipecat_voice/prompts/<variant>.py`. Each module exposes `AGENT_SYSTEM_PROMPT` as a complete string (no `str.format` placeholders). The CLI flag `--prompt-variant <name>` selects one for the run; default is `baseline` (verbatim from tau2's `LLMAgent`).

Adding a new variant is one file plus one line in `pipecat_voice/prompts/__init__.py`.

### Eval iteration loop

For each failure mode we want to address:

1. Run `baseline` (no variant) on tasks 0–4 with `--write-summary --check all`. The summary's *Failure-mode checks* table tells us which checks failed per task.
2. Write a prompt variant that targets the failure mode. Don't touch the pipeline.
3. Rerun with `--prompt-variant <name> --out data/runs/<fix>`. The summary's `Per task` table shows new reward; the *Failure-mode checks* table shows which checks now pass.
4. Capture before/after numbers in the summary and the run's `SUMMARY.md`.

### Baseline (tasks 0–4, real stack, VAD on)

Run with the real stack: Parakeet STT, Chatterbox TTS, MiniMax M2.7 for both the agent and the user simulator, Silero VAD on. 300s conversation timeout, single trial per task (`data/runs/baseline_m27`).

Scoring is local-only: tau2's ENV + ACTION + COMMUNICATE evaluators merged like `ALL`, with `strict_replay=False` (voice transcripts garble ids, so strict replay aborts instead of grading). NL_ASSERTIONS are excluded — the NL judge defaults to gpt-4.1 and needs `OPENAI_API_KEY`, which this harness doesn't carry (MiniMax key only).

| task | reason_for_call (short) | gold write | reward | actions | checks (auth/no_tool/prem) |
|---|---|---|---|---|---|
| 0 | exchange two items in order #W2378156 | `exchange_delivered_order_items` | 0.0 | 0/5 | P / **F** / P |
| 1 | exchange two items in order #W2378156 (variant) | `exchange_delivered_order_items` | 0.0 | 2/5 | **F** / P / P |
| 2 | count t-shirt options + return cleaner + headphones | `return_delivered_order_items` | 0.0 | 1/11 | P / P / P |
| 3 | count t-shirt options + modify pending small t-shirts | `modify_pending_order_items` | 0.0 | 10/12 | **F** / P / P |
| 4 | count t-shirt options + modify pending t-shirts | `modify_pending_order_items` (×2) | 0.0 | 10/13 | **F** / P / P |

All five start with `find_user_id_by_name_zip` as the first read, then several reads, then one or more writes. Dominant failure: the agent re-calls the same lookup 5–40× (auth loop) and never advances; task 0 never calls any tool at all. Tasks 3/4 match 10 actions yet fail on DB state — the repeated calls break sequence alignment and arg fidelity.

### Prompt variants applied so far (task 1 A/B)

| variant | what changed | actions | auth_loop check |
|---|---|---|---|
| baseline (300s) | — | 2/5 | FAIL (33 lookups) |
| v1 (180s) | auth-once + act-don't-stall + read-back rules | 4/5 | FAIL (11× same lookup) |
| v2 (180s) | v1 + explicit "after user id, next action MUST be order/product" | 4/5 | FAIL (loop moved to `get_order_details` 4×) |
| v3 (180s) | v2 + confirm-before-call + retry-at-most-once | 0/5 | n/a (no calls — over-correction into passivity) |
| v2 (300s rerun) | same as v2 | 1/5 | FAIL (10× lookup) |

Reading: v1/v2 lift action completion 2/5 → 4/5 in less time; v2 pushes the stall downstream (auth → order stage) but doesn't clear it; v3 proves the boundary — confirmation requirements without a forcing function stall the agent into pure text. The 300s v2 rerun scoring 1/5 against the 180s 4/5 shows run-to-run voice variance (STT luck, sampling) dominates small deltas — any claimed improvement needs multi-trial stats before it counts.

### Tradeoffs

- **Voice round-trip is in the eval loop.** A STT misrecognition or TTS mispronunciation can mask or cause an agent failure, so a "prompt fix" might not move reward until STT/TTS quality is also addressed. The trace events on `voice_trace.jsonl` distinguish a content failure from a transcript failure.
- **Closed loop means the user simulator and the agent share model behaviour.** Switching the user simulator to a faster / cheaper model could shift timing without changing the agent. We keep them on the same model deliberately.
- **MiniMax M2.7 only.** Other MiniMax variants are easy to swap via `cfg.agent_model` / `cfg.user_model`, but mixing them risks confounding the prompt-vs-model signal. (M2.7 standard replaced High-Speed; it emits thinking blocks, which Pipecat handles, at higher first-token latency.)
- **NL assertions unscored.** Tasks whose `reward_basis` includes NL_ASSERTION are graded on ENV+ACTION+COMMUNICATE only, noted in `reward_info.info`. Full `ALL` scoring needs an OpenAI key for the judge.
- **Single-trial cells.** With voice-channel variance this high (same prompt scored 4/5 then 1/5 on rerun), every number above is indicative, not significant. `--num-trials N` exists; the eval half should report mean/variance per cell before any improvement claim sticks.

### Future improvements

- **Multi-trial evaluation.** `--num-trials N` already runs N trials per task; the eval half should report mean / variance per cell.
- **More domains.** Retail-only today. Adding airline or telecom is one `--domain` flag and a separate tau2 dataset.
- **Streamlit viewer.** A separate read-only viewer in `viewer/` walks `data/runs/` and renders transcripts, reward breakdown, and a matplotlib trace timeline per sim.
- **Streaming eval.** The harness already runs in real time; an `--asr-debug` flag that captures raw audio chunks alongside STT output would let us quantify STT error rates separately from agent error rates.

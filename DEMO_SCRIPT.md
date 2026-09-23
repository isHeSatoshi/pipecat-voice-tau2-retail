# Under-5-Minute Demo Script

## 0:00–0:30 — Problem and baseline

- Show Tau-bench retail task 1 and the Pipecat/Tau2 architecture.
- Open `data/runs/baseline_m27/behavior_report.json`.
- State: 0/5 strict success, 100% auth-loop failure, 100% write-protocol failure, 80% tool-argument failure.

## 0:30–1:20 — Root causes

Show these three artifacts:

1. `baseline_run.err`: Silero VAD initialization exception.
2. `baseline_m27/task_2/.../trajectory.json`: empty and incorrect tool arguments.
3. Old `voice_trace.jsonl`: lifecycle events only, no STT/LLM/tool/audio evidence.

Explain that prompt tuning alone was measuring broken infrastructure and an empty tool schema.

## 1:20–2:20 — Three evals and interventions

- `src/pipecat_voice/eval/checks.py`: `auth_loop`, `tool_argument_integrity`, `write_protocol`.
- `src/pipecat_voice/prompts/v4.py`: explicit voice-agent state machine.
- `src/pipecat_voice/tau2/tool_bridge.py`: actual Tau2 schemas, argument validation, serialized exact-call guard.
- `src/pipecat_voice/transport/virtual_transport.py`: initialized Silero and audio capture.
- `src/pipecat_voice/pipelines/pipecat_adapters.py`: one complete LLM response becomes one TTS waveform.

## 2:20–3:40 — Before and after

- Open baseline task 1 in the viewer.
- Open `working_v4_task1_final` in the viewer.
- Show transcript, trace timeline, and both audio players.
- State the measured result: auth loop PASS, argument integrity PASS, write protocol PASS; the thermostat exchange tool returned `exchange requested`.

## 3:40–4:30 — Trade-offs and next experiment

- Local Tau2 scoring excludes NL assertions.
- One focused run is evidence, not significance.
- Next experiment: three paired seeds on tasks 0, 1, and held-out task 5, plus a text-only control.
- Natural completion detection now ends “That’s all I needed” as `user_stop`; strict Tau2 reward still requires NL judging.

## 4:30–4:50 — Reproduce

Show:

```powershell
python -m pipecat_voice.cli analyze --run data/runs/baseline_m27
python -m pipecat_voice.cli run --domain retail --task 1 --prompt-variant v4 --out data/runs/v4_guard --write-summary --check all
streamlit run viewer/app.py
```

## 4:50–5:00 — Close

State the main result: the focused v4 + guard conversation authenticates, grounds all IDs, enforces confirmation, successfully writes the retail exchange, passes all three behavior evals, and now has natural termination detection.

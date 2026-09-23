# pipecat_voice viewer

Read-only Streamlit viewer for the pipecat_voice eval harness.

It walks `data/runs/` (the folder the runner writes to) and renders
transcripts, reward breakdowns, and trace timelines for each simulation.
It never writes anything to disk.

## Install

```powershell
uv pip install streamlit
```

## Run

From `D:\Project\infer_task\pipecat_voice\`:

```powershell
streamlit run viewer/app.py
```

Streamlit opens a browser tab at `http://localhost:8501`. The sidebar has:

- **Runs root** — folder to scan (defaults to `data/runs` next to the viewer).
- **Select run** — which run folder to open.
- **Select sim** — which task/sim under that run.
- **Compare runs** — multi-select; enables the side-by-side reward view.

## What you see

- **Run detail** view: per-sim header (reward, db match, action matched, termination, duration), tabs for transcript, reward breakdown, and a matplotlib trace timeline.
- **Compare runs** view: table of `(run, task, reward, …)` rows and a horizontal bar chart of reward per task across runs.

## Layout assumption

The viewer reads the artefacts written by `pipecat_voice.tau2.runner.Tau2EvalRunner`:

```
data/runs/<run_name>/
    summary.json
    task_<id>/sim_<uuid>/
        trajectory.json
        voice_trace.jsonl
```

Missing files are tolerated: a sim with no `trajectory.json` shows
"no messages"; a run with no `summary.json` is shown as `(partial)` with
whatever task/sim output exists on disk.

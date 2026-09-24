# Pipecat voice viewer

A read-only Streamlit viewer for trajectories saved by the `pipecat_voice` harness.

The viewer scans `data/runs/`, lets you select a run and simulation, and renders:

- task, seed, local reward, DB match, action count, termination, and duration;
- transcript and tool-call history;
- local reward and action-check breakdowns;
- conversation-relative trace timeline;
- agent, user, and mixed conversation audio;
- cross-run comparison;
- timed live refresh for in-progress artifacts.

It never starts an evaluation and never writes to the runs directory.

## Install and run

From the repository root:

```bash
uv pip install -e ".[viewer]"
streamlit run viewer/app.py
```

Streamlit opens at `http://localhost:8501`. The sidebar contains the runs root, run and simulation selectors, live-stream toggle and interval, and comparison selector.

## Data layout

```text
data/runs/<run_name>/
    summary.json
    task_<id>/sim_<id>/
        trajectory.json
        voice_trace.jsonl
        audio_segments.json
        agent_audio.wav
        user_audio.wav
        conversation.wav
        audio/
```

Missing files are tolerated. A simulation without `trajectory.json` is shown as partial, and a run without `summary.json` is still available for artifact inspection.

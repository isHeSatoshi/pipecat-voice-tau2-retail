"""Streamlit app for browsing pipecat_voice eval runs.

Read-only viewer. Picks runs and sims from a sidebar and renders:

- **Run list** (sidebar): every folder under ``data/runs`` with a
  ``summary.json`` or any ``task_*/sim_*/`` output (the latter are marked
  ``(partial)``); checkbox to add it to the comparison set.
- **Compare runs** view: side-by-side table of (run, task, reward, …).
- **Single sim** view: transcript with role-aware chat bubbles (user /
  assistant / tool result), reward breakdown, and a matplotlib trace
  timeline.

Run with::

    streamlit run viewer/app.py

The viewer re-scans the runs folder on every rerun (which happens when
the user clicks **Refresh** or when **Auto-refresh** is enabled), so
runs started in another terminal appear without restarting Streamlit.
"""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

import streamlit as st

# Make ``viewer`` importable when Streamlit runs ``app.py`` directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viewer import render as R  # noqa: E402
from viewer import run_loader  # noqa: E402

DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent.parent / "data" / "runs"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="pipecat_voice viewer",
    page_icon="🎙️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Loaders (no caching: we want every rerun to read fresh disk state)
# ---------------------------------------------------------------------------


def _runs_root() -> Path:
    """Sidebar text input for the runs root (defaults to ``data/runs``)."""
    st.session_state.setdefault("viewer_runs_root", str(DEFAULT_RUNS_ROOT))
    return Path(
        st.sidebar.text_input(
            "Runs root",
            key="viewer_runs_root",
            help="Folder containing one subfolder per run (each with summary.json).",
        )
    )


def _run_mtime(run: run_loader.RunSummary) -> float:
    files = [run.path / "summary.json"]
    files.extend(run.path.glob("task_*/sim_*/trajectory.json"))
    files.extend(run.path.glob("task_*/sim_*/voice_trace.jsonl"))
    files.extend(run.path.glob("task_*/sim_/*.wav"))
    return max((path.stat().st_mtime for path in files if path.exists()), default=0.0)


def _load_runs(runs_root: Path) -> list[run_loader.RunSummary]:
    runs = [
        run_loader.load_summary(path) for path in run_loader.discover_runs(runs_root)
    ]
    return sorted(runs, key=_run_mtime, reverse=True)


def _sim_mtime(sim: run_loader.SimView) -> float:
    """Most-recent mtime across the sim's files (for the freshness badge)."""
    files = [sim.path / "trajectory.json", sim.path / "voice_trace.jsonl"]
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


def _selected_run(
    summaries: list[run_loader.RunSummary],
) -> run_loader.RunSummary | None:
    if not summaries:
        return None
    names = [summary.name for summary in summaries]
    current = st.session_state.get("viewer_selected_run")
    if current not in names:
        st.session_state["viewer_selected_run"] = names[0]
    choice = st.sidebar.selectbox(
        "Select run",
        names,
        key="viewer_selected_run",
        help="Choose a completed or in-progress run.",
    )
    return next((s for s in summaries if s.name == choice), summaries[0])


def _available_sims(run: run_loader.RunSummary) -> list[run_loader.SimView]:
    """Return simulations in newest-first order without rendering widgets."""
    sims = run_loader.discover_sims(run.path)
    sims.sort(key=_sim_mtime, reverse=True)
    return sims


def _resolve_sim(
    run: run_loader.RunSummary,
    preferred_sim_id: str | None = None,
) -> run_loader.SimView | None:
    """Load a preferred sim, falling back to the newest available sim."""
    sims = _available_sims(run)
    if not sims:
        return None
    sim_ids = {s.sim_id for s in sims}
    sim_id = preferred_sim_id or st.session_state.get("viewer_selected_sim")
    if sim_id not in sim_ids:
        sim_id = sims[0].sim_id
    sim = next(s for s in sims if s.sim_id == sim_id)
    return run_loader.load_sim(sim)


def _sim_seed(sim) -> int | None:
    """Read a seed without assuming a hot-reloaded SimView class has it."""
    seed = getattr(sim, "seed", None)
    if seed is not None:
        return seed
    trajectory = getattr(sim, "trajectory", None) or {}
    seed = trajectory.get("seed")
    return None if seed is None else int(seed)


def _selected_sim(run: run_loader.RunSummary) -> run_loader.SimView | None:
    sims = _available_sims(run)
    if not sims:
        st.info("No simulations under this run.")
        return None
    sim_ids = [s.sim_id for s in sims]
    labels = {
        s.sim_id: (
            f"task {s.task_id} · seed {_sim_seed(s) if _sim_seed(s) is not None else '—'} "
            f"· sim {s.sim_id[:8]}"
        )
        for s in sims
    }
    current = st.session_state.get("viewer_selected_sim")
    if current not in sim_ids:
        st.session_state["viewer_selected_sim"] = sim_ids[0]
    choice = st.sidebar.selectbox(
        "Select sim",
        sim_ids,
        format_func=lambda sim_id: labels[sim_id],
        key="viewer_selected_sim",
        help="Choose a simulation. The seed is read from its artifacts.",
    )
    sim = next((s for s in sims if s.sim_id == choice), sims[0])
    return run_loader.load_sim(sim)


def _compare_rows(runs: list[run_loader.RunSummary]) -> list[dict]:
    rows: list[dict] = []
    for r in runs:
        for sim in run_loader.discover_sims(r.path):
            sim = run_loader.load_sim(sim)
            rows.append(run_loader.reward_table_row(sim, r.name))
    rows.sort(key=lambda x: (x["task_id"], x["run"]))
    return rows


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _freshness_badge(sim: run_loader.SimView) -> str:
    """Short label: 'live' if files were updated in the last 30s."""
    mt = _sim_mtime(sim)
    if not mt:
        return "—"
    age = time.time() - mt
    if age < 5:
        return "🔴 live (just updated)"
    if age < 30:
        return f"🟡 updated {int(age)}s ago"
    if age < 600:
        return f"🟢 updated {int(age / 60)}m ago"
    return f"⚪ updated {int(age / 60)}m ago"


def page_compare(runs: list[run_loader.RunSummary]) -> None:
    st.header("Compare runs")
    if len(runs) < 2:
        st.info("Pick at least two runs in the sidebar to compare.")
        return
    rows = _compare_rows(runs)
    if not rows:
        st.info("No simulations across the selected runs.")
        return
    st.dataframe(rows, width="stretch", hide_index=True)

    # Reward-by-task chart.
    import matplotlib.pyplot as plt

    task_ids = sorted({r["task_id"] for r in rows})
    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.5 * len(task_ids) + 2)))
    width = 0.8 / max(1, len(runs))
    for i, r in enumerate(runs):
        per_task = {
            row["task_id"]: (row["reward"] or 0.0)
            for row in rows
            if row["run"] == r.name
        }
        ys = [per_task.get(t, 0.0) for t in task_ids]
        ax.barh(
            [i + j * width for j in range(len(task_ids))],
            ys,
            height=width,
            label=r.name,
        )
    ax.set_yticks([i + width * (len(task_ids) - 1) / 2 for i in range(len(runs))])
    ax.set_yticklabels([r.name for r in runs])
    ax.set_xlabel("reward")
    ax.set_xlim(0, 1.05)
    ax.set_title("Reward by task across runs")
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def page_run(run: run_loader.RunSummary) -> None:
    st.header(f"Run: `{run.name}`")
    if not run.complete:
        st.warning(
            "Run looks incomplete — no `summary.json` (interrupted before the "
            "CLI finished). Showing the task/sim output found on disk."
        )
    cfg = run.config or {}
    if cfg:
        config_items = [
            f"{k}={v}"
            for k, v in cfg.items()
            if k
            in {
                "domain",
                "stt",
                "tts",
                "agent_llm",
                "user_llm",
                "minimax_api_base",
                "split",
                "seed",
                "seeds",
                "max_conversation_seconds",
            }
        ]
        st.caption(" · ".join(config_items))

    # Per-run results table.
    results = run.results or []
    if results:
        st.markdown("**Results**")
        st.dataframe(
            [
                {
                    "task_id": r.get("task_id"),
                    "trial": r.get("trial"),
                    "seed": r.get("seed"),
                    "local_reward": r.get("local_reward"),
                    "strict_reward": r.get("reward")
                    if r.get("strict_reward_available", True)
                    else None,
                    "db_match": r.get("db_match"),
                    "actions": f"{r.get('actions_matched', 0)}/{r.get('actions_total', 0)}",
                    "termination": r.get("termination_reason"),
                    "duration_s": round(r.get("duration_seconds", 0) or 0, 1),
                    "trajectory": r.get("trajectory_path"),
                }
                for r in results
            ],
            width="stretch",
            hide_index=True,
        )

    sim = _selected_sim(run)
    if sim is None:
        return

    st.divider()
    st.caption(_freshness_badge(sim))
    R.render_run_header(sim)

    tab_t, tab_r, tab_tr, tab_a = st.tabs(
        ["Transcript", "Reward", "Trace timeline", "Audio"]
    )
    with tab_t:
        R.render_transcript(sim)
    with tab_r:
        R.render_reward(sim)
    with tab_tr:
        R.render_trace_timeline(sim)
    with tab_a:
        for label, filename in (
            ("Agent audio", "agent_audio.wav"),
            ("User audio", "user_audio.wav"),
        ):
            audio_path = sim.path / filename
            if audio_path.exists():
                st.caption(label)
                st.audio(str(audio_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _sidebar_controls(runs: list[run_loader.RunSummary]) -> tuple[list[str], bool, int]:
    """Sidebar controls. Returns (compare_names, auto_refresh, refresh_secs)."""
    st.sidebar.divider()
    # Refresh: forces a full rerun (re-reads files, repicks sims, etc.).
    if st.sidebar.button("🔄 Refresh now", help="Re-scan runs/sims from disk."):
        st.rerun()
    auto_refresh = st.sidebar.checkbox(
        "Live stream (auto-refresh)",
        value=False,
        help="Reload run artifacts every N seconds while an eval is in progress.",
    )
    refresh_secs = st.sidebar.number_input(
        "Refresh interval (s)",
        min_value=2,
        max_value=120,
        value=10,
        step=1,
        disabled=not auto_refresh,
    )
    if auto_refresh:
        st.info(
            "Live stream is enabled. The selected run view will refresh on the "
            "interval above."
        )

    compare_names = st.sidebar.multiselect(
        "Compare runs",
        [r.name for r in runs],
        default=[],
        help="Select 2+ runs to enable the Compare view.",
    )
    return compare_names, auto_refresh, int(refresh_secs)


def _live_view(
    runs_root: Path,
    *,
    page: str,
    compare_names: list[str],
    refresh_secs: int,
) -> None:
    """Render the selected page inside the timed live-refresh fragment."""
    @st.fragment(run_every=timedelta(seconds=refresh_secs), key="live_view")
    def _render_live_view() -> None:
        fresh_runs = _load_runs(runs_root)
        if not fresh_runs:
            st.warning("No runs are available yet.")
            return

        selected_name = st.session_state.get("viewer_selected_run")
        fresh_run = next(
            (run for run in fresh_runs if run.name == selected_name),
            fresh_runs[0],
        )
        st.caption(
            f"Live update · {len(fresh_runs)} runs · selected {fresh_run.name}"
        )

        if page == "Run detail":
            page_run(fresh_run)
            return

        fresh_compare_runs = [run for run in fresh_runs if run.name in compare_names]
        page_compare(fresh_compare_runs or fresh_runs[:2])

    _render_live_view()


def main() -> None:
    runs_root = _runs_root()
    runs = _load_runs(runs_root)

    if not runs:
        st.warning(f"No runs found under {runs_root}. Run the eval first.")
        st.sidebar.button("🔄 Refresh now")
        return

    compare_names, auto_refresh, refresh_secs = _sidebar_controls(runs)
    compare_runs = [r for r in runs if r.name in compare_names]

    selected = _selected_run(runs)
    if selected is None:
        return

    page = st.sidebar.radio("View", ["Run detail", "Compare runs"], index=0)
    if page == "Run detail":
        if auto_refresh:
            _live_view(
                runs_root,
                page=page,
                compare_names=compare_names,
                refresh_secs=refresh_secs,
            )
        else:
            page_run(selected)
    else:
        if auto_refresh:
            _live_view(
                runs_root,
                page=page,
                compare_names=compare_names,
                refresh_secs=refresh_secs,
            )
        else:
            page_compare(compare_runs or runs[:2])


if __name__ == "__main__":
    main()
else:
    # Streamlit may import the script under a different module name in
    # some launch modes. Do not hide rendering errors; surface them.
    main()

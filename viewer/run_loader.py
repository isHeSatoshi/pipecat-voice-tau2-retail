"""Read-only loaders for pipecat_voice run artefacts.

All paths are constructed from a single root (the run directory under
``data/runs/``). This module never writes anything; it only reads JSON and
JSONL files and returns plain Python data structures.

File layout expected (created by ``tau2.runner.Tau2EvalRunner``):

    data/runs/<run_name>/
        summary.json
        task_<id>/
            sim_<uuid>/
                trajectory.json
                voice_trace.jsonl

Anything missing is treated as empty / None; the viewer should still
render for the fields that are present.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Default runs root, used when the Streamlit sidebar input is empty.
DEFAULT_RUNS_ROOT = Path(
    os.environ.get(
        "PIPECAT_VOICE_RUNS_ROOT",
        str(Path(__file__).resolve().parent.parent / "data" / "runs"),
    )
)


# ---------------------------------------------------------------------------
# Data classes (plain, no Pydantic — viewer is a leaf consumer)
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Top-level summary.json for one run."""

    name: str
    path: Path
    config: dict[str, Any]
    results: list[dict[str, Any]]
    raw: dict[str, Any]
    # False when summary.json is missing (run interrupted before the CLI
    # wrote it). Sims found on disk are still viewable.
    complete: bool = True


@dataclass
class SimView:
    """One simulation folder under a run."""

    run_name: str
    task_id: str
    sim_id: str
    path: Path
    trajectory: Optional[dict[str, Any]] = None
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    audio_manifest: dict[str, Any] = field(default_factory=dict)
    conversation_audio: Optional[Path] = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _has_sims(run_dir: Path) -> bool:
    """True when the dir holds any ``task_*/sim_*/`` output."""
    try:
        return any(
            sim.is_dir() and sim.name.startswith("sim_")
            for task in run_dir.iterdir()
            if task.is_dir() and task.name.startswith("task_")
            for sim in task.iterdir()
        )
    except OSError:
        return False


def discover_runs(runs_root: Path) -> list[Path]:
    """Return sorted run directories with results.

    A run qualifies with ``summary.json`` (complete) or with any
    ``task_*/sim_*/`` output (interrupted before the summary was written).
    """
    if not runs_root.exists():
        return []
    out: list[Path] = []
    for p in sorted(runs_root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir():
            continue
        if (p / "summary.json").exists() or _has_sims(p):
            out.append(p)
    return out


def load_summary(run_dir: Path) -> RunSummary:
    """Load a run's ``summary.json`` (with defensive defaults)."""
    summary_path = run_dir / "summary.json"
    raw: dict[str, Any] = {}
    complete = summary_path.exists()
    if complete:
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = {}
            complete = False
    return RunSummary(
        name=run_dir.name,
        path=run_dir,
        config=raw.get("config", {}) or {},
        results=list(raw.get("results", []) or []),
        raw=raw,
        complete=complete,
    )


def discover_sims(run_dir: Path) -> list[SimView]:
    """Walk the ``task_<id>/sim_<uuid>/`` tree under a run directory."""
    out: list[SimView] = []
    if not run_dir.exists():
        return out
    for task_dir in sorted(run_dir.iterdir(), key=lambda x: x.name.lower()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        for sim_dir in sorted(task_dir.iterdir(), key=lambda x: x.name.lower()):
            if not sim_dir.is_dir() or not sim_dir.name.startswith("sim_"):
                continue
            out.append(
                SimView(
                    run_name=run_dir.name,
                    task_id=task_dir.name.replace("task_", "", 1),
                    sim_id=sim_dir.name.replace("sim_", "", 1),
                    path=sim_dir,
                )
            )
    return out


def load_sim(sim: SimView) -> SimView:
    """Populate ``trajectory`` and ``trace_events`` for a SimView (in-place)."""
    traj_path = sim.path / "trajectory.json"
    if traj_path.exists():
        try:
            sim.trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sim.trajectory = None
    trace_path = sim.path / "voice_trace.jsonl"
    if trace_path.exists():
        events: list[dict[str, Any]] = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        sim.trace_events = events
    manifest_path = sim.path / "audio_segments.json"
    if manifest_path.exists():
        try:
            sim.audio_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sim.audio_manifest = {}
    conversation_name = sim.audio_manifest.get("conversation")
    if conversation_name:
        conversation_path = sim.path / conversation_name
        if conversation_path.exists():
            sim.conversation_audio = conversation_path
    trajectory_messages = (sim.trajectory or {}).get("messages") or []
    for segment in sim.audio_manifest.get("segments") or []:
        message_index = segment.get("message_index")
        segment_path = sim.path / str(segment.get("path", ""))
        if (
            isinstance(message_index, int)
            and 0 <= message_index < len(trajectory_messages)
            and segment_path.exists()
        ):
            trajectory_messages[message_index]["audio_path"] = str(segment_path)
    return sim


# ---------------------------------------------------------------------------
# Convenience extractors
# ---------------------------------------------------------------------------


def messages(sim: SimView) -> list[dict[str, Any]]:
    """Return the canonical message list from the trajectory.

    Filters out ``system`` messages (the system prompt is part of the
    policy snapshot, not the trajectory).
    """
    traj = sim.trajectory or {}
    out: list[dict[str, Any]] = []
    for m in traj.get("messages", []) or []:
        if m.get("role") == "system":
            continue
        out.append(m)
    return out


def reward_summary(sim: SimView) -> dict[str, Any]:
    """Compact summary of ``reward_info`` for table views."""
    ri = (sim.trajectory or {}).get("reward_info") or {}
    db = ri.get("db_check") or {}
    actions = ri.get("action_checks") or []
    matched = sum(1 for a in actions if a.get("action_match"))
    total = len(actions)
    env_asserts = ri.get("env_assertions") or []
    env_pass = sum(1 for a in env_asserts if a.get("met"))
    info = ri.get("info") or {}
    strict_available = info.get("strict_reward_available")
    if strict_available is None:
        strict_available = "NL_ASSERTION" not in (ri.get("reward_basis") or [])
    partial_reward = info.get("partial_reward")
    if partial_reward is None and strict_available:
        partial_reward = ri.get("reward")
    return {
        "reward": ri.get("reward"),
        "partial_reward": partial_reward,
        "strict_reward_available": strict_available,
        "db_match": db.get("db_match"),
        "actions_total": total,
        "actions_matched": matched,
        "env_assertions_total": len(env_asserts),
        "env_assertions_passed": env_pass,
    }


def config_comparison(runs: list[RunSummary]) -> dict[str, dict[str, Any]]:
    """Build a dict-of-dicts for side-by-side run config comparison."""
    out: dict[str, dict[str, Any]] = {}
    for r in runs:
        out[r.name] = dict(r.config)
    return out


def reward_table_row(sim: SimView, run_name: str) -> dict[str, Any]:
    """One row per (run, task) for the compare-runs table."""
    r = reward_summary(sim)
    traj = sim.trajectory or {}
    return {
        "run": run_name,
        "task_id": sim.task_id,
        "sim_id": sim.sim_id,
        "reward": r["reward"],
        "local_reward": r["partial_reward"]
        if not r["strict_reward_available"]
        else r["reward"],
        "strict_reward_available": r["strict_reward_available"],
        "termination": traj.get("termination_reason"),
        "duration": traj.get("duration"),
        "actions_matched": r["actions_matched"],
        "actions_total": r["actions_total"],
        "db_match": r["db_match"],
    }

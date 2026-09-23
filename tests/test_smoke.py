"""End-to-end smoke tests for pipecat_voice.

The single-task smoke runs the dummy harness against tau2's retail task 0
and asserts:

1. The CLI exits 0.
2. ``trajectory.json`` is written under ``<out>/task_0/<sim_id>/``.
3. ``reward_info`` is present (reward may be 0; we just check the shape).
4. ``voice_trace.jsonl`` exists with at least a ``meta`` event.

The protocol-swap smoke is covered by ``test_swap_bounds.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_help() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "pipecat_voice.cli", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0
    assert "pipecat-voice" in out.stdout.lower() or "voice eval" in out.stdout.lower()


def test_dummy_smoke_one_task(tmp_path: Path) -> None:
    """Run retail task 0 with all dummy services; verify outputs."""
    out_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "-m",
        "pipecat_voice.cli",
        "run",
        "--task",
        "0",
        "--stt",
        "dummy",
        "--tts",
        "dummy",
        "--agent-llm",
        "dummy",
        "--user-llm",
        "dummy",
        "--max-seconds",
        "8",
        "--out",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    # The CLI must exit 0; stderr may contain DEBUG output.
    assert proc.returncode == 0, f"CLI failed: {proc.stderr[-2000:]}"

    # Check outputs.
    task_dir = out_dir / "task_0"
    assert task_dir.exists(), f"No task_0 dir produced: {list(out_dir.iterdir())}"
    sim_dirs = [p for p in task_dir.iterdir() if p.is_dir()]
    assert len(sim_dirs) == 1, f"Expected 1 sim dir, got {len(sim_dirs)}"
    sim_dir = sim_dirs[0]

    trajectory = sim_dir / "trajectory.json"
    assert trajectory.exists(), f"Missing trajectory.json: {list(sim_dir.iterdir())}"
    data = json.loads(trajectory.read_text(encoding="utf-8"))
    assert data["task_id"] == "0"
    assert data["termination_reason"] in {"user_stop", "agent_stop", "timeout"}
    assert data["reward_info"] is not None
    assert "db_check" in data["reward_info"]
    assert "action_checks" in data["reward_info"]

    trace = sim_dir / "voice_trace.jsonl"
    assert trace.exists()
    events = [
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    types = [e.get("type") for e in events]
    assert "meta" in types
    assert "run_start" in types
    assert "run_end" in types

    summary = out_dir / "summary.json"
    assert summary.exists()
    sdata = json.loads(summary.read_text(encoding="utf-8"))
    assert sdata["results"][0]["task_id"] == "0"


def test_dummy_two_tasks_in_one_run(tmp_path: Path) -> None:
    """Run two retail tasks back to back; verify both produced outputs."""
    out_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "-m",
        "pipecat_voice.cli",
        "run",
        "--tasks",
        "0",
        "1",
        "--stt",
        "dummy",
        "--tts",
        "dummy",
        "--agent-llm",
        "dummy",
        "--user-llm",
        "dummy",
        "--max-seconds",
        "6",
        "--out",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, f"CLI failed: {proc.stderr[-2000:]}"

    assert (out_dir / "task_0").exists()
    assert (out_dir / "task_1").exists()


def test_user_stop_detector_accepts_natural_completion() -> None:
    from pipecat_voice.pipelines.user_pipeline import StopOnUserSignalProcessor

    assert StopOnUserSignalProcessor._should_stop("Great, that's all I needed.")
    assert StopOnUserSignalProcessor._should_stop("###STOP###")
    assert not StopOnUserSignalProcessor._should_stop("Please proceed.")


def test_help_run_subcommand() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "pipecat_voice.cli", "run", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0
    assert "--task" in out.stdout
    assert "--stt" in out.stdout
    assert "--tts" in out.stdout
    assert "--agent-llm" in out.stdout

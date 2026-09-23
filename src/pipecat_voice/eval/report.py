from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import RewardType

from pipecat_voice.eval import run_all_checks
from pipecat_voice.tau2.environment import load_tasks_filtered


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def analyze_run(
    run_dir: Path, *, domain: str = "retail", split: str = "base"
) -> dict[str, Any]:
    paths = sorted(run_dir.glob("task_*/sim_*/trajectory.json"))
    if not paths:
        raise ValueError(f"No trajectory files found under {run_dir}")
    task_ids = sorted({path.parts[-3].removeprefix("task_") for path in paths})
    tasks = {
        task.id: task for task in load_tasks_filtered(domain, split, task_ids=task_ids)
    }
    records: list[dict[str, Any]] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        simulation = SimulationRun.model_validate(raw)
        task = tasks.get(simulation.task_id)
        if task is None:
            continue
        outcome = run_all_checks(simulation, task)
        reward = simulation.reward_info.reward if simulation.reward_info else None
        messages = simulation.messages or []
        max_user_run = 0
        current_user_run = 0
        for message in messages:
            if getattr(message, "role", "") == "user":
                current_user_run += 1
                max_user_run = max(max_user_run, current_user_run)
            else:
                current_user_run = 0
        failed_checks = [result.name for result in outcome.results if not result.passed]
        basis = set(task.evaluation_criteria.reward_basis)
        strict_reward_available = RewardType.NL_ASSERTION not in basis
        records.append(
            {
                "task_id": simulation.task_id,
                "simulation_id": simulation.id,
                "path": str(path),
                "reward": reward,
                "strict_success": bool(
                    reward is not None
                    and reward >= 0.999
                    and strict_reward_available
                    and simulation.termination_reason == "user_stop"
                    and not failed_checks
                ),
                "duration": simulation.duration or 0.0,
                "messages": len(messages),
                "timeout": simulation.termination_reason == "timeout",
                "fragmented_user_turn": max_user_run >= 2,
                "failed_checks": failed_checks,
                "checks": {
                    result.name: {"passed": result.passed, "message": result.message}
                    for result in outcome.results
                },
            }
        )
    check_names = list(records[0]["checks"]) if records else []
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "simulations": len(records),
        "mean_reward": _mean([r["reward"] for r in records if r["reward"] is not None]),
        "strict_success_rate": _mean([float(r["strict_success"]) for r in records]),
        "timeout_rate": _mean([float(r["timeout"]) for r in records]),
        "fragmented_turn_rate": _mean(
            [float(r["fragmented_user_turn"]) for r in records]
        ),
        "mean_duration_seconds": _mean([r["duration"] for r in records]),
        "mean_messages": _mean([r["messages"] for r in records]),
        "behavior_failure_rates": {
            name: _mean([float(not r["checks"][name]["passed"]) for r in records])
            for name in check_names
        },
        "records": records,
    }

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from tau2.data_model.simulation import SimulationRun

from pipecat_voice.observability.tau2_bridge import run_evaluator_local
from pipecat_voice.tau2.environment import load_tasks_filtered


def _completion_duration(sim_dir: Path) -> float:
    trace_path = sim_dir / "voice_trace.jsonl"
    offsets = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in {"tts_stop", "stt_result", "tool_result"}:
                offsets.append(float(event.get("t_offset_ms", 0.0)) / 1000)
    return max(offsets, default=0.0)


def _has_natural_completion(simulation: SimulationRun) -> bool:
    phrases = ("that's all i needed", "that is all i needed")
    for message in simulation.messages or []:
        content = getattr(message, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if getattr(message, "role", "") == "user" and any(
            phrase in str(content).lower() for phrase in phrases
        ):
            return True
    return False


def rescore(path: Path, *, domain: str, split: str, write: bool) -> dict:
    simulation = SimulationRun.model_validate_json(path.read_text(encoding="utf-8"))
    if (
        _has_natural_completion(simulation)
        and simulation.termination_reason == "timeout"
    ):
        duration = _completion_duration(path.parent)
        simulation.termination_reason = "user_stop"
        simulation.duration = duration
        start = datetime.fromisoformat(simulation.start_time)
        simulation.end_time = (start + timedelta(seconds=duration)).isoformat()
    tasks = load_tasks_filtered(domain, split, task_ids=[simulation.task_id])
    reward_info = run_evaluator_local(simulation, tasks[0])
    dump = reward_info.model_dump(mode="json")
    if write:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["termination_reason"] = simulation.termination_reason
        data["duration"] = simulation.duration
        data["end_time"] = simulation.end_time
        data["reward_info"] = dump
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    actions = dump.get("action_checks") or []
    return {
        "path": str(path),
        "termination": simulation.termination_reason,
        "duration": simulation.duration,
        "strict_reward": dump.get("reward"),
        "partial_reward": (dump.get("info") or {}).get("partial_reward"),
        "strict_reward_available": (dump.get("info") or {}).get(
            "strict_reward_available"
        ),
        "db_match": (dump.get("db_check") or {}).get("db_match"),
        "actions_matched": sum(1 for action in actions if action.get("action_match")),
        "actions_total": len(actions),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--split", default="base")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.run.glob("task_*/sim_*/trajectory.json"))
    results = [
        rescore(path, domain=args.domain, split=args.split, write=args.write)
        for path in paths
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

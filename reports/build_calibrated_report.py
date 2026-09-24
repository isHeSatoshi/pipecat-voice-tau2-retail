"""Build derived tables and charts for the calibrated evaluation bundle.

This script reads an existing run directory and writes only derived report files.
It never edits the raw trajectories, traces, audio, or summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAINING_TASKS = {"0", "1", "2", "3", "4", "6", "7"}
TEST_TASKS = {"5", "9", "12"}
CHECKS = ("auth_loop", "tool_argument_integrity", "write_protocol")


def split_for(task_id: str) -> str:
    return "train" if task_id in TRAINING_TASKS else "test" if task_id in TEST_TASKS else "other"


def as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def bool_int(value: Any) -> int | None:
    if value is True:
        return 1
    if value is False:
        return 0
    return None


def sim_id_for(result: dict[str, Any], batch_root: Path) -> str | None:
    raw = result.get("trajectory_path")
    if not raw:
        return None
    raw_path = Path(str(raw))
    if raw_path.parent.name.startswith("sim_"):
        return raw_path.parent.name
    matches = list(batch_root.glob(f"task_{result.get('task_id')}/sim_*/trajectory.json"))
    if len(matches) == 1:
        return matches[0].parent.name
    return None


def locate_sim(batch_root: Path, result: dict[str, Any]) -> Path | None:
    sim_id = sim_id_for(result, batch_root)
    if sim_id is None:
        return None
    path = batch_root / f"task_{result.get('task_id')}" / sim_id
    return path if path.is_dir() else None


def result_checks(result: dict[str, Any]) -> dict[str, bool]:
    return {c["name"]: bool(c.get("passed")) for c in result.get("checks", []) if c.get("name") in CHECKS}


def representative_key(item: tuple[dict[str, Any], Path | None]) -> tuple[Any, ...]:
    result, _ = item
    local = as_float(result.get("local_reward"))
    db = int(result.get("db_match") is True)
    total = result.get("actions_total") or 0
    matched = result.get("actions_matched") or 0
    fraction = matched / total if total else 0.0
    non_timeout = int(result.get("termination_reason") != "timeout")
    return (local, db, fraction, non_timeout, -as_float(result.get("duration_seconds"), 1e9))


def make_tables(batch_root: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads((batch_root / "summary.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[tuple[dict[str, Any], Path | None]]] = defaultdict(list)
    for index, result in enumerate(summary.get("results", []), start=1):
        task_id = str(result.get("task_id"))
        sim_dir = locate_sim(batch_root, result)
        grouped[task_id].append((result, sim_dir))
        checks = result_checks(result)
        total = result.get("actions_total") or 0
        matched = result.get("actions_matched") or 0
        trajectory_path = sim_dir / "trajectory.json" if sim_dir else None
        audio_available = bool(
            sim_dir
            and (sim_dir / "agent_audio.wav").exists()
            and (sim_dir / "user_audio.wav").exists()
        )
        rows.append(
            {
                "attempt": index,
                "task_id": task_id,
                "split": split_for(task_id),
                "trial": result.get("trial"),
                "seed": result.get("seed"),
                "sim_id": sim_dir.name if sim_dir else "",
                "local_reward": as_float(result.get("local_reward")),
                "strict_reward": as_float(result.get("reward")),
                "strict_reward_available": bool(result.get("strict_reward_available")),
                "db_match": result.get("db_match"),
                "actions_matched": matched,
                "actions_total": total,
                "action_fraction": matched / total if total else 0.0,
                "termination": result.get("termination_reason"),
                "duration_s": as_float(result.get("duration_seconds")),
                "auth_loop": checks.get("auth_loop"),
                "tool_argument_integrity": checks.get("tool_argument_integrity"),
                "write_protocol": checks.get("write_protocol"),
                "all_checks_pass": all(checks.values()) if len(checks) == len(CHECKS) else False,
                "infrastructure_error": result.get("infrastructure_error") or "",
                "artifact_status": (
                    "recovered_no_audio"
                    if sim_dir and not audio_available
                    else "complete"
                    if trajectory_path and trajectory_path.exists()
                    else "incomplete"
                ),
                "relative_trajectory": str(Path("data") / trajectory_path.relative_to(batch_root)) if trajectory_path and trajectory_path.exists() else "",
            }
        )

    representatives = []
    for task_id in sorted(grouped, key=int):
        items = grouped[task_id]
        result, sim_dir = max(items, key=representative_key)
        task_rows = [r for r in rows if r["task_id"] == task_id]
        checks = result_checks(result)
        total = result.get("actions_total") or 0
        matched = result.get("actions_matched") or 0
        representatives.append(
            {
                "task_id": task_id,
                "split": split_for(task_id),
                "attempts": len(task_rows),
                "local_success_attempts": sum(r["local_reward"] == 1.0 for r in task_rows),
                "db_match_attempts": sum(r["db_match"] is True for r in task_rows),
                "full_action_attempts": sum(
                    r["actions_total"] > 0 and r["actions_matched"] == r["actions_total"] for r in task_rows
                ),
                "representative_sim": sim_dir.name if sim_dir else "",
                "representative_local_reward": as_float(result.get("local_reward")),
                "representative_db_match": result.get("db_match"),
                "representative_actions": f"{matched}/{total}" if total else "0/0",
                "representative_termination": result.get("termination_reason"),
                "representative_duration_s": as_float(result.get("duration_seconds")),
                "auth_loop": checks.get("auth_loop"),
                "tool_argument_integrity": checks.get("tool_argument_integrity"),
                "write_protocol": checks.get("write_protocol"),
            }
        )

    artifacts = []
    for sim_dir in sorted(batch_root.glob("task_*/sim_*"), key=lambda p: (int(p.parent.name[5:]), p.name)):
        trajectory = sim_dir / "trajectory.json"
        trace = sim_dir / "voice_trace.jsonl"
        artifacts.append(
            {
                "task_id": sim_dir.parent.name.removeprefix("task_"),
                "sim_id": sim_dir.name.removeprefix("sim_"),
                "trajectory": trajectory.exists(),
                "trace": trace.exists(),
                "agent_audio": (sim_dir / "agent_audio.wav").exists(),
                "user_audio": (sim_dir / "user_audio.wav").exists(),
                "conversation_audio": (sim_dir / "conversation.wav").exists(),
                "audio_manifest": (sim_dir / "audio_segments.json").exists(),
                "status": "complete" if trajectory.exists() and trace.exists() else "metadata_only",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "attempts.csv", rows)
    _write_csv(out_dir / "task_summary.csv", representatives)
    _write_csv(out_dir / "artifacts.csv", artifacts)
    _write_csv(
        out_dir / "checks.csv",
        [
            {
                "attempt": row["attempt"],
                "task_id": row["task_id"],
                "split": row["split"],
                "auth_loop": row["auth_loop"],
                "tool_argument_integrity": row["tool_argument_integrity"],
                "write_protocol": row["write_protocol"],
            }
            for row in rows
        ],
    )
    return rows, representatives, artifacts, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_charts(rows: list[dict[str, Any]], tasks: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = [row["task_id"] for row in tasks]
    x = np.arange(len(tasks))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    local = [row["representative_local_reward"] for row in tasks]
    db = [bool_int(row["representative_db_match"]) if row["representative_db_match"] is not None else np.nan for row in tasks]
    axes[0].bar(x, local, color=["#2a9d8f" if value == 1.0 else "#e76f51" for value in local])
    axes[0].set_ylim(0, 1.15)
    axes[0].set_xticks(x, task_ids)
    axes[0].set_xlabel("Retail task")
    axes[0].set_ylabel("Local reward")
    axes[0].set_title("Representative local reward by task")
    axes[0].axhline(1.0, color="#333", linewidth=0.8, linestyle="--")
    axes[0].text(0.02, 0.92, "NL judge excluded", transform=axes[0].transAxes, fontsize=8)
    axes[1].bar(x, [0 if value is None else value for value in db], color=["#2a9d8f" if value == 1 else "#e76f51" if value == 0 else "#bbb" for value in db])
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xticks(x, task_ids)
    axes[1].set_xlabel("Retail task")
    axes[1].set_ylabel("DB match (1 = yes)")
    axes[1].set_title("Representative database match")
    for index, value in enumerate(db):
        if np.isnan(value):
            axes[1].text(index, 0.05, "N/A", ha="center", va="bottom", fontsize=8)
    fig.suptitle("v4_calibrated_batch: task-level outcomes", fontsize=14)
    fig.savefig(out_dir / "task_outcomes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    for task_id in sorted({row["task_id"] for row in rows}, key=int):
        subset = [row for row in rows if row["task_id"] == task_id]
        jitter = np.linspace(-0.16, 0.16, len(subset)) if len(subset) > 1 else np.array([0.0])
        colors = {"timeout": "#e76f51", "user_stop": "#2a9d8f", "agent_stop": "#457b9d"}
        for row, offset in zip(subset, jitter):
            ax.scatter(int(task_id) + offset, row["local_reward"], s=90, color=colors.get(row["termination"], "#555"), edgecolor="white", linewidth=0.8)
    ax.set_xticks(sorted({int(row["task_id"]) for row in rows}))
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel("Retail task")
    ax.set_ylabel("Attempt local reward")
    ax.set_title("All recorded attempts: local reward and termination")
    ax.legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=label, markersize=8) for label, color in [("user stop", "#2a9d8f"), ("agent stop", "#457b9d"), ("timeout", "#e76f51")]], loc="lower right", frameon=False)
    fig.savefig(out_dir / "attempt_outcomes.png", dpi=180)
    plt.close(fig)

    pass_counts = {name: sum(row[name] is True for row in rows) for name in CHECKS}
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    names = ["Authentication loop", "Tool arguments", "Write protocol"]
    values = [pass_counts[name] / len(rows) for name in CHECKS]
    bars = ax.barh(names, values, color=["#2a9d8f", "#e9c46a", "#457b9d"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pass rate across recorded attempts")
    ax.set_title("Behavior-check results")
    for bar, value, count in zip(bars, values, pass_counts.values()):
        ax.text(min(value + 0.02, 0.96), bar.get_y() + bar.get_height() / 2, f"{count}/{len(rows)}", va="center", fontsize=9)
    fig.savefig(out_dir / "check_pass_rates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    durations = [row["duration_s"] for row in rows]
    colors = {"timeout": "#e76f51", "user_stop": "#2a9d8f", "agent_stop": "#457b9d"}
    ax.bar(np.arange(len(rows)), durations, color=[colors.get(row["termination"], "#555") for row in rows])
    ax.set_xticks(np.arange(len(rows)), [f"T{row['task_id']}·{row['attempt']}" for row in rows], rotation=75, ha="right")
    ax.set_ylabel("Conversation duration (s)")
    ax.set_title("Duration by recorded attempt")
    fig.savefig(out_dir / "attempt_durations.png", dpi=180)
    plt.close(fig)

    matrix = np.full((len(tasks), len(CHECKS)), np.nan)
    for i, task in enumerate(tasks):
        for j, name in enumerate(CHECKS):
            value = task[name]
            if value is True:
                matrix[i, j] = 1
            elif value is False:
                matrix[i, j] = 0
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CHECKS)), ["Auth loop", "Tool args", "Write protocol"])
    ax.set_yticks(range(len(tasks)), task_ids)
    ax.set_title("Representative behavior checks")
    for i in range(len(tasks)):
        for j in range(len(CHECKS)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, "pass" if matrix[i, j] == 1 else "fail", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pass (1) / fail (0)")
    fig.savefig(out_dir / "representative_check_matrix.png", dpi=180)
    plt.close(fig)


def make_provenance(batch_root: Path, rows: list[dict[str, Any]], tasks: list[dict[str, Any]], artifacts: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    raw_summary = json.loads((batch_root / "summary.json").read_text(encoding="utf-8"))
    config = raw_summary.get("config", {})
    payload = {
        "generated_from": "data/runs/v4_calibrated_batch/summary.json",
        "raw_results_modified": False,
        "summary_result_count": len(rows),
        "simulation_directory_count": len(artifacts),
        "unique_task_count": len({row["task_id"] for row in rows}),
        "task_ids": sorted({row["task_id"] for row in rows}, key=int),
        "declared_split": config.get("split"),
        "observed_splits": sorted({row["split"] for row in rows}),
        "local_reward_success_attempts": sum(row["local_reward"] == 1.0 for row in rows),
        "db_match_attempts": sum(row["db_match"] is True for row in rows),
        "timeout_attempts": sum(row["termination"] == "timeout" for row in rows),
        "strict_reward_available_count": sum(row["strict_reward_available"] for row in rows),
        "infrastructure_error_count": sum(bool(row["infrastructure_error"]) for row in rows),
        "incomplete_simulation_directories": [row for row in artifacts if row["status"] != "complete"],
        "caveats": [
            "The raw root SUMMARY.md is stale: it reports 14 rows while summary.json contains 19 result records.",
            "The root config declares split=test, while the bundle also contains training task IDs 0-7.",
            "Task 0 and task 1 summary paths point to byte-identical copies from other local run directories; the copies are retained in this bundle.",
            "Task 2 is a recovered transcript with no audio files.",
            "Three metadata-only simulation directories are present but are not counted in summary.json results.",
        ],
        "config_snapshot": config,
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(batch_root: Path, out_dir: Path, rows: list[dict[str, Any]], tasks: list[dict[str, Any]], artifacts: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    success = sum(row["local_reward"] == 1.0 for row in rows)
    db = sum(row["db_match"] is True for row in rows)
    timeouts = sum(row["termination"] == "timeout" for row in rows)
    all_checks = sum(row["all_checks_pass"] for row in rows)
    lines = [
        "# Calibrated batch report",
        "",
        "This report is derived from `data/runs/v4_calibrated_batch/summary.json` and its retained simulation artifacts. The raw summary, trajectories, traces, and audio are not modified.",
        "",
        "## Executive summary",
        "",
        f"- **Recorded attempts:** {len(rows)} across {len(tasks)} unique retail task IDs ({len(set(row['task_id'] for row in rows))} unique in the raw record).",
        f"- **Local reward 1.0:** {success}/{len(rows)} recorded attempts ({success / len(rows):.1%}).",
        f"- **DB match:** {db}/{len(rows)} recorded attempts ({db / len(rows):.1%}).",
        f"- **Timeouts:** {timeouts}/{len(rows)} recorded attempts ({timeouts / len(rows):.1%}).",
        f"- **All three behavior checks passed:** {all_checks}/{len(rows)} recorded attempts.",
        "- **Strict Tau2 reward availability:** unavailable for all 19 records. The recorded strict field is 0 because the harness excluded the `NL_ASSERTION` dimension; the NL judge was not configured.",
        "",
        "The local reward is a useful deterministic local score, not a substitute for full strict Tau2 reward. The strict evaluator requires an OpenAI NL judge (`gpt-4.1` in the harness), while this run used a MiniMax key and deliberately ran only local ENV, ACTION, and COMMUNICATE evaluators.",
        "",
        "## Visual summaries",
        "",
        "![Representative task outcomes](charts/task_outcomes.png)",
        "",
        "![All attempt outcomes](charts/attempt_outcomes.png)",
        "",
        "![Behavior-check pass rates](charts/check_pass_rates.png)",
        "",
        "![Attempt durations](charts/attempt_durations.png)",
        "",
        "![Representative check matrix](charts/representative_check_matrix.png)",
        "",
        "## Task-level view",
        "",
        "A representative is selected for display by highest local reward, then DB match, then action fraction, then non-timeout termination. This is a presentation choice, not a replacement for the full attempt table.",
        "",
        "| Task | Split | Attempts | Local 1.0 attempts | DB match attempts | Representative | Local | DB | Actions | Termination | Auth | Args | Write |",
        "|---:|:---:|---:|---:|---:|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in tasks:
        lines.append(
            f"| {row['task_id']} | {row['split']} | {row['attempts']} | {row['local_success_attempts']} | {row['db_match_attempts']} | `{row['representative_sim']}` | {row['representative_local_reward']:.1f} | {row['representative_db_match'] if row['representative_db_match'] is not None else 'N/A'} | {row['representative_actions']} | {row['representative_termination']} | {_mark(row['auth_loop'])} | {_mark(row['tool_argument_integrity'])} | {_mark(row['write_protocol'])} |"
        )
    lines += [
        "",
        "## Honest interpretation",
        "",
        "- **Task 5 is the clear test-task failure.** The bundle contains four recorded task-5 attempts, all with local reward 0.0. Three timed out, and the non-timeout attempt ended with a failed authentication path and no successful expected write. It is not a successful task under any local metric. Task 5 is also inside the earlier 0-7 development slice, so it is not a clean held-out task boundary despite its final test-set designation.",
        "- **Task 9 is a qualified local success.** Its representative attempt has local reward 1.0, DB match, and 6/6 actions. Its `tool_argument_integrity` check still fails because a recorded authentication call returned `User not found`; therefore the local reward should not be read as a clean behavior-check pass.",
        "- **Task 12 is a DB success but a protocol failure.** Its representative attempt has local reward 1.0, DB match, and 4/5 actions, but the write fails because PayPal was used instead of the order's original payment method. The `write_protocol` check correctly fails.",
        "- **Training-task outcomes are mixed across attempts.** The bundle contains successful local representatives for training tasks 0, 1, 2, 3, 4, 6, and 7, but several representatives still fail one or more behavior checks. Task 3 and task 6 contain both successful and unsuccessful attempts, so selecting a success without showing the attempt count would overstate stability.",
        "- **The bundle is mixed provenance.** The root config declares `split=test`, while the bundle includes training task IDs 0-7 and the selected test IDs 5, 9, and 12. Task 5 is therefore present in both the 0-7 development slice and the reported test set, as the user’s final selection specifies. The raw `SUMMARY.md` is stale at 14 rows; `summary.json` is the authoritative 19-record source for this report. Task 2 is a recovered transcript without audio, and three metadata-only simulation directories are retained for completeness.",
        "- **No infrastructure error was recorded** for the 19 summary records, but strict NL scoring was not attempted. The report separates local outcome, DB/action checks, and strict-reward availability.",
        "",
        "## Reproduce the derived report",
        "",
        "```powershell",
        "python reports/build_calibrated_report.py .\\data\\runs\\v4_calibrated_batch .\\reports\\v4_calibrated_batch",
        "```",
        "",
        "The command above only reads raw artifacts and rewrites derived tables/charts under `reports/v4_calibrated_batch/`.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mark(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "N/A"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows, tasks, artifacts, summary = make_tables(args.batch, args.output)
    make_charts(rows, tasks, args.output / "charts")
    make_provenance(args.batch, rows, tasks, artifacts, summary, args.output)
    write_markdown(args.batch, args.output, rows, tasks, artifacts, summary)
    print(f"wrote {len(rows)} attempt rows, {len(tasks)} task rows, and {len(artifacts)} artifact rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

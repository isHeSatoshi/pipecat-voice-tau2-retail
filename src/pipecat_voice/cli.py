"""CLI entry point.

Usage
-----

::

    # 1 task, dummy impls, no network or GPU required
    python -m pipecat_voice.cli run --task 0 \
        --stt dummy --tts dummy --agent-llm dummy --user-llm dummy

    # 1 task, real stack
    python -m pipecat_voice.cli run --task 0 \
        --stt parakeet --tts chatterbox \
        --agent-llm minimax --user-llm minimax

    # Batch over the tau2 retail base split
    python -m pipecat_voice.cli run --split base --num-trials 1 \
        --max-tasks 5 --out data/runs/baseline

Every run writes to ``<out>/task_<id>/sim_<uuid>/`` containing
``trajectory.json`` and ``voice_trace.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from pipecat_voice.config import load_config
from pipecat_voice.tau2.environment import load_tasks_filtered
from pipecat_voice.tau2.runner import Tau2EvalRunner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipecat-voice", description="Pipecat + tau2 retail voice eval."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser(
        "run", help="Run one or more tau2 tasks through the voice harness."
    )
    run.add_argument(
        "--task", type=str, default=None, help="Single tau2 task id to run."
    )
    run.add_argument(
        "--tasks", type=str, nargs="*", default=None, help="Multiple task ids to run."
    )
    run.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Tau-bench domain (retail, airline, telecom, mock).",
    )
    run.add_argument(
        "--split",
        type=str,
        default=None,
        help="Task split name(s), comma-separated (e.g. base, train, test).",
    )
    run.add_argument(
        "--append",
        action="store_true",
        help="Append results to an existing run summary instead of replacing it.",
    )
    run.add_argument(
        "--num-trials",
        type=int,
        choices=(1,),
        default=1,
        help="Number of trials per task; exactly one is supported.",
    )
    run.add_argument(
        "--max-tasks", type=int, default=None, help="Cap on number of tasks to run."
    )

    run.add_argument(
        "--stt", type=str, default=None, help="STT impl: parakeet | whisper | dummy."
    )
    run.add_argument(
        "--tts",
        type=str,
        default=None,
        help="TTS impl: chatterbox | elevenlabs | dummy.",
    )
    run.add_argument(
        "--agent-llm",
        type=str,
        default=None,
        help="Agent LLM impl: minimax | anthropic | dummy.",
    )
    run.add_argument(
        "--user-llm",
        type=str,
        default=None,
        help="User LLM impl: minimax | anthropic | dummy.",
    )
    run.add_argument("--agent-model", type=str, default=None, help="Agent model id.")
    run.add_argument(
        "--user-model", type=str, default=None, help="User simulator model id."
    )
    run.add_argument("--seed", type=int, default=None, help="Random seed.")
    run.add_argument("--out", type=str, default=None, help="Output directory.")
    run.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Max conversation duration (seconds).",
    )
    run.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="PCM sample rate on the audio bus.",
    )
    run.add_argument(
        "--no-vad", action="store_true", help="Disable Silero VAD (debug only)."
    )
    run.add_argument(
        "--write-summary",
        action="store_true",
        help="After the run finishes, write/refresh a human-readable SUMMARY.md next to summary.json.",
    )
    run.add_argument(
        "--prompt-variant",
        type=str,
        default=None,
        help="Name of the agent system-prompt variant to load (e.g. baseline, v1, v2). See pipecat_voice.prompts.",
    )
    run.add_argument(
        "--check",
        type=str,
        default=None,
        help=(
            "After the run, evaluate the named failure-mode check(s) "
            "(comma-separated, or 'all') over every trajectory and "
            "append to SUMMARY.md. Example: --check auth_loop"
        ),
    )

    analyze = sub.add_parser(
        "analyze", help="Analyze saved trajectories without rerunning models."
    )
    analyze.add_argument("--run", type=str, nargs="+", required=True)
    analyze.add_argument("--domain", type=str, default=None)
    analyze.add_argument("--split", type=str, default=None)
    analyze.add_argument("--out", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "analyze":
        return _cmd_analyze(args)
    return 0


def _cmd_analyze(args) -> int:
    from pipecat_voice.eval.report import analyze_run

    reports = []
    for path in args.run:
        run_dir = Path(path)
        config = {}
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                config = json.loads(summary_path.read_text(encoding="utf-8")).get(
                    "config", {}
                )
            except (OSError, json.JSONDecodeError):
                config = {}
        reports.append(
            analyze_run(
                run_dir,
                domain=args.domain or config.get("domain", "retail"),
                split=args.split or config.get("split", "base"),
            )
        )
    payload = reports[0] if len(reports) == 1 else {"runs": reports}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _cmd_run(args) -> int:
    cfg = load_config()
    overrides = {}
    if args.stt is not None:
        overrides["stt_impl"] = args.stt
    if args.tts is not None:
        overrides["tts_impl"] = args.tts
    if args.agent_llm is not None:
        overrides["agent_llm_impl"] = args.agent_llm
    if args.user_llm is not None:
        overrides["user_llm_impl"] = args.user_llm
    if args.domain is not None:
        overrides["domain"] = args.domain
    if args.agent_model is not None:
        overrides["agent_model"] = args.agent_model
    if args.user_model is not None:
        overrides["user_model"] = args.user_model
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.out is not None:
        overrides["out_dir"] = Path(args.out)
    if args.max_seconds is not None:
        overrides["max_conversation_seconds"] = args.max_seconds
    if args.sample_rate is not None:
        overrides["sample_rate"] = args.sample_rate
    cfg = cfg.merge_cli(**overrides)

    if not cfg.minimax_api_key and "minimax" in (cfg.agent_llm_impl, cfg.user_llm_impl):
        logger.error(
            "ANTHROPIC_API_KEY (the MiniMax-compatible key) is not set. Add it "
            "to tau2-bench/.env or pipecat_voice/.env, or pass --agent-llm "
            "dummy for a local-only run."
        )
        return 2

    # Optional --prompt-variant override on the agent system prompt.
    pv = getattr(args, "prompt_variant", None)
    if pv:
        from pipecat_voice.prompts import load_agent_prompt

        try:
            cfg.agent_system_prompt_override = load_agent_prompt(pv)
            cfg.prompt_variant = pv
            logger.info(f"Loaded prompt variant '{pv}' for agent system prompt.")
        except KeyError as e:
            logger.error(str(e))
            return 2

    # Resolve task list.
    if args.task:
        task_ids = [args.task]
    elif args.tasks:
        task_ids = list(args.tasks)
    else:
        task_ids = []

    split = args.split if args.split is not None else cfg.task_split
    split_names = [part.strip() for part in split.split(",") if part.strip()]
    if not split_names:
        logger.error("At least one task split is required")
        return 1
    cfg.task_split = ",".join(split_names)
    tasks = []
    remaining = args.max_tasks
    for split_name in split_names:
        loaded = load_tasks_filtered(
            cfg.domain,
            split_name,
            task_ids=task_ids,
            max_tasks=remaining,
        )
        tasks.extend(loaded)
        if remaining is not None:
            remaining = max(0, remaining - len(loaded))
            if remaining == 0:
                break
    if not tasks:
        logger.error(f"No tasks matched task_ids={task_ids!r} split={split!r}")
        return 1
    found_task_ids = {str(task.id) for task in tasks}
    missing_task_ids = sorted(set(task_ids) - found_task_ids)
    if missing_task_ids:
        logger.error(
            f"Unknown task ids for domain={cfg.domain} split={split}: "
            f"{missing_task_ids}"
        )
        return 1

    logger.info(
        f"Running {len(tasks)} tasks with stt={cfg.stt_impl}, tts={cfg.tts_impl}, "
        f"agent_llm={cfg.agent_llm_impl}, user_llm={cfg.user_llm_impl}, "
        f"out={cfg.out_dir}"
    )

    base_seed = cfg.seed
    runner = Tau2EvalRunner(cfg=cfg, trace_dir=cfg.out_dir, enable_vad=not args.no_vad)

    results = []
    for task in tasks:
        logger.info(f"[trial 1/1] task={task.id}")
        try:
            res = runner.run(task)
        except Exception as e:
            logger.exception(f"Task {task.id} failed: {e}")
            res = {"task_id": task.id, "error": str(e)}
        res["trial"] = 1
        res["seed"] = base_seed
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))

    # Write a summary (best-effort: if a previous run's summary file is
    # still locked on Windows, we still return the results).
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.out_dir / "summary.json"
    existing_results = []
    if args.append and summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_results = list(existing.get("results") or [])
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read existing summary for append: {e}")
    else:
        try:
            if summary_path.exists():
                summary_path.unlink()
        except OSError:
            pass
    summary_results = [*existing_results, *results]
    try:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "domain": cfg.domain,
                        "split": split,
                        "stt": cfg.stt_impl,
                        "tts": cfg.tts_impl,
                        "agent_llm": cfg.agent_llm_impl,
                        "user_llm": cfg.user_llm_impl,
                        "minimax_api_base": cfg.minimax_api_base,
                        "agent_model": cfg.agent_model,
                        "user_model": cfg.user_model,
                        "prompt_variant": cfg.prompt_variant,
                        "seed": base_seed,
                        "max_conversation_seconds": cfg.max_conversation_seconds,
                        "vad_enabled": not args.no_vad,
                        "tool_execution_policy": "single_serialized_exact_signature",
                    },
                    "results": summary_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Wrote summary: {summary_path}")
    except OSError as e:
        logger.warning(f"Could not write summary.json: {e}")

    if args.write_summary:
        try:
            check_names = _resolve_check_names(args.check)
            _write_run_summary_md(
                cfg.out_dir,
                cfg,
                split,
                summary_results,
                check_names=check_names,
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not write SUMMARY.md: {e}")
    if any(
        result.get("error") or result.get("infrastructure_error") for result in results
    ):
        return 1
    return 0


def _resolve_check_names(spec: Optional[str]) -> Optional[list[str]]:
    """Resolve ``--check`` value into a list of check names."""
    if spec is None:
        return None
    spec = spec.strip()
    if spec in {"", "none", "off"}:
        return None
    if spec == "all":
        from pipecat_voice.eval import CHECKS

        return list(CHECKS.keys())
    return [n.strip() for n in spec.split(",") if n.strip()]


def _write_run_summary_md(
    out_dir: Path,
    cfg,
    split: str,
    results: list[dict],
    check_names: Optional[list[str]] = None,
) -> Path:
    """Render a markdown summary alongside ``summary.json``.

    The point is to give a reader the headline numbers and per-task status
    without having to open the JSON. The trajectory files are still the
    source of truth — the viewer reads them directly.
    """
    md_path = out_dir / "SUMMARY.md"
    n = len(results)
    rewards = [
        r.get("reward") for r in results if isinstance(r.get("reward"), (int, float))
    ]
    avg_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    passed = sum(1 for r in rewards if r >= 0.999)
    duration = sum((r.get("duration_seconds") or 0) for r in results)

    lines: list[str] = []
    lines.append(f"# Run summary — `{out_dir.name}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    for k in (
        "domain",
        "split",
        "stt_impl",
        "tts_impl",
        "agent_llm_impl",
        "user_llm_impl",
        "agent_model",
        "user_model",
        "minimax_api_base",
        "max_conversation_seconds",
        "seed",
    ):
        v = getattr(cfg, k, None)
        if v not in (None, ""):
            lines.append(f"| {k} | `{v}` |")
    lines.append(f"| prompt_variant | `{cfg.prompt_variant}` |")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Tasks: **{n}**")
    lines.append(f"- Mean reward: **{avg_reward:.3f}**")
    lines.append(f"- Pass count (reward == 1.0): **{passed}/{n}**")
    lines.append(f"- Total wall time: **{duration:.1f} s**")
    lines.append("")
    lines.append("## Per task")
    lines.append("")
    lines.append("| task | trial | reward | termination | duration (s) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        lines.append(
            f"| {r.get('task_id')} | {r.get('trial')} | {r.get('reward')} | {r.get('termination_reason')} | {round((r.get('duration_seconds') or 0), 1)} |"
        )
    lines.append("")

    # Optional failure-mode check section.
    if check_names:
        lines.append("## Failure-mode checks")
        lines.append("")
        try:
            check_section = _render_check_section(
                out_dir, results, check_names, domain=cfg.domain, split=split
            )
            if check_section:
                lines.extend(check_section)
            else:
                lines.append("_No trajectories found; nothing to check._")
        except Exception as e:  # pragma: no cover
            lines.append(f"_Check evaluation failed: {e}_")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _render_check_section(
    out_dir: Path,
    results: list[dict],
    check_names: list[str],
    *,
    domain: str = "retail",
    split: str = "base",
) -> list[str]:
    """Render per-task failure-mode check rows for SUMMARY.md.

    Returns markdown lines (without trailing newline) describing the
    pass/fail outcome of every requested check on every trajectory.
    Returns ``[]`` if no trajectory files are present.
    """
    from pipecat_voice.eval import run_all_checks

    rows: list[str] = []
    header_done = False
    for r in results:
        traj_path = r.get("trajectory_path")
        task_id = r.get("task_id")
        if not traj_path:
            continue
        # Resolve relative paths against out_dir.
        raw_path = Path(traj_path)
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(out_dir / raw_path)
            candidates.append(out_dir / raw_path.name)
            if task_id is not None:
                candidates.extend(out_dir.glob(f"task_{task_id}/sim_*/trajectory.json"))
        p = next(
            (candidate.resolve() for candidate in candidates if candidate.exists()),
            None,
        )
        if p is None:
            continue
        try:
            from tau2.data_model.simulation import SimulationRun

            sim_run = SimulationRun.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Re-load the task.
        task = None
        try:
            from pipecat_voice.tau2.environment import load_tasks_filtered

            loaded = load_tasks_filtered(
                domain,
                split,
                task_ids=[task_id] if task_id else None,
            )
            task = loaded[0] if loaded else None
        except Exception:
            task = None
        if task is None:
            continue

        outcome = run_all_checks(sim_run, task)
        if not header_done:
            cells = ["task", "check", "passed", "message"]
            rows.append("| " + " | ".join(cells) + " |")
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            header_done = True
        for cr in outcome.results:
            if check_names and cr.name not in check_names:
                continue
            rows.append(
                f"| {task_id} | {cr.name} | "
                f"{'✅' if cr.passed else '❌'} | {cr.message.replace('|', '/')} |"
            )
    return rows


if __name__ == "__main__":
    sys.exit(main())

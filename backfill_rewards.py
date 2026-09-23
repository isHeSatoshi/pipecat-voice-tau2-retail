"""Re-score saved trajectories with the local (judge-free) evaluator.

Usage: python backfill_rewards.py [--write]
  --write: update trajectory.json reward_info in place (default: report only).
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tau2.data_model.simulation import SimulationRun

from pipecat_voice.tau2.environment import load_tasks_filtered
from pipecat_voice.observability.tau2_bridge import run_evaluator_local

WRITE = "--write" in sys.argv
tasks = {t.id: t for t in load_tasks_filtered("retail", "base",
                                              task_ids=["0", "1", "2", "3", "4"])}

for tid in ["0", "1", "2", "3", "4"]:
    paths = glob.glob(f"data/runs/baseline_m27/task_{tid}/*/trajectory.json")
    if not paths:
        print(f"task {tid}: no trajectory")
        continue
    p = paths[0]
    data = json.load(open(p))
    try:
        run = SimulationRun.model_validate(data)
        ri = run_evaluator_local(run, tasks[tid])
        dump = ri.model_dump(mode="json")
        acts = dump.get("action_checks") or []
        matched = sum(1 for a in acts if a.get("action_match"))
        print(f"task {tid}: reward={dump.get('reward')} "
              f"actions={matched}/{len(acts)} "
              f"db_match={(dump.get('db_check') or {}).get('db_match')} "
              f"breakdown={dump.get('reward_breakdown')}")
        if WRITE:
            data["reward_info"] = dump
            json.dump(data, open(p, "w"), indent=2, ensure_ascii=False)
            print(f"  wrote {p}")
    except Exception as e:
        print(f"task {tid}: SCORING FAILED: {type(e).__name__}: {str(e)[:300]}")

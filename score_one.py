"""Score one saved trajectory: 3 behavior checks + local reward."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tau2.data_model.simulation import SimulationRun

from pipecat_voice.tau2.environment import load_tasks_filtered
from pipecat_voice.eval import run_all_checks
from pipecat_voice.observability.tau2_bridge import run_evaluator_local

run_dir, task_id = sys.argv[1], sys.argv[2]
task = load_tasks_filtered("retail", "base", task_ids=[task_id])[0]
p = glob.glob(f"{run_dir}/task_{task_id}/*/trajectory.json")[0]
data = json.load(open(p))
run = SimulationRun.model_validate(data)
outcome = run_all_checks(run, task)
print("checks:", " ".join(f"{r.name}={'PASS' if r.passed else 'FAIL'}" for r in outcome.results))
for r in outcome.results:
    if not r.passed:
        print(f"  {r.name}: {r.message[:200]}")
try:
    ri = run_evaluator_local(run, task)
    d = ri.model_dump(mode="json")
    acts = d.get("action_checks") or []
    m = sum(1 for a in acts if a.get("action_match"))
    print(f"reward={d.get('reward')} actions={m}/{len(acts)} "
          f"db_match={(d.get('db_check') or {}).get('db_match')} breakdown={d.get('reward_breakdown')}")
    if "--write" in sys.argv:
        data["reward_info"] = d
        json.dump(data, open(p, "w"), indent=2, ensure_ascii=False)
        print("  wrote", p)
except Exception as e:
    print("SCORING FAILED:", type(e).__name__, str(e)[:300])

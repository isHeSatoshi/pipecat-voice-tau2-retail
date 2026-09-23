"""Run the 3 behavior checks against saved baseline trajectories."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tau2.data_model.simulation import SimulationRun

from pipecat_voice.tau2.environment import load_tasks_filtered
from pipecat_voice.eval import run_all_checks

tasks = {t.id: t for t in load_tasks_filtered("retail", "base",
                                              task_ids=["0", "1", "2", "3", "4"])}

for tid in ["0", "1", "2", "3", "4"]:
    paths = glob.glob(f"data/runs/baseline_m27/task_{tid}/*/trajectory.json")
    if not paths:
        print(f"task {tid}: no trajectory")
        continue
    data = json.load(open(paths[0]))
    run = SimulationRun.model_validate(data)
    outcome = run_all_checks(run, tasks[tid])
    flags = " ".join(f"{r.name}={'PASS' if r.passed else 'FAIL'}" for r in outcome.results)
    print(f"task {tid}: {flags}")
    for r in outcome.results:
        if not r.passed:
            print(f"    {r.name}: {r.message[:160]}")

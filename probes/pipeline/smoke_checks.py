"""Smoke-test eval checks against a real trajectory."""
import sys; sys.path.insert(0, '.')
from pipecat_voice.eval import run_all_checks
from tau2.data_model.simulation import SimulationRun
from pathlib import Path
from viewer.run_loader import DEFAULT_RUNS_ROOT
from pipecat_voice.tau2.environment import load_tasks_filtered

sim_dir = next(DEFAULT_RUNS_ROOT.glob('real_coherent/task_0/sim_*'))
print('sim dir:', sim_dir)
sim_run = SimulationRun.model_validate_json(Path(sim_dir, 'trajectory.json').read_text(encoding='utf-8'))
print('msgs:', len(sim_run.messages or []))
tasks = load_tasks_filtered('retail', 'base', task_ids=[sim_run.task_id])
task = tasks[0]
print('task:', task.id)

outcome = run_all_checks(sim_run, task)
for r in outcome.results:
    flag = "PASS" if r.passed else "FAIL"
    print(f'  {r.name}: {flag} - {r.message}')

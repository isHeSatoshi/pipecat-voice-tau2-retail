"""Evaluation checks for pipecat_voice runs.

A check is a small predicate over a finished ``SimulationRun`` plus its
``Task`` that flags a specific failure mode the agent is known to fall
into. The runner reports pass/fail per task; the eval half of the
harness uses these to decide whether a prompt variant is an
improvement.

Adding a new check:

1. Define a function with signature
   ``def check_X(sim_run: SimulationRun, task: Task) -> CheckResult``.
2. Register it in ``CHECKS`` at module bottom.
3. The CLI flag ``--check <name>`` will run it.

The checks are *pure* — no I/O, no LLM calls, no global state. They
read messages and tool calls from ``sim_run.messages`` and decide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task

from pipecat_voice.eval.checks import CheckResult  # noqa: E402

CheckFn = Callable[[SimulationRun, Task], CheckResult]


@dataclass
class CheckOutcome:
    """Outcome of running all checks against one (task, sim) pair."""

    task_id: str
    sim_id: str
    results: List[CheckResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed]


def run_all_checks(sim_run: SimulationRun, task: Task) -> CheckOutcome:
    """Run every registered check against one simulation."""
    results = [fn(sim_run, task) for fn in CHECKS.values()]
    sim_id = sim_run.id or "unknown"
    return CheckOutcome(task_id=task.id, sim_id=sim_id, results=results)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CHECKS: Dict[str, CheckFn] = {}


def register(name: str, fn: CheckFn) -> None:
    if name in CHECKS:
        raise ValueError(f"Check {name!r} already registered")
    CHECKS[name] = fn


def _register_defaults() -> None:
    # Lazy imports so this module stays import-time cheap.
    from pipecat_voice.eval import checks as _checks

    register("auth_loop", _checks.check_auth_loop)
    register("tool_argument_integrity", _checks.check_tool_argument_integrity)
    register("write_protocol", _checks.check_write_protocol)


_register_defaults()


__all__ = [
    "CheckResult",
    "CheckOutcome",
    "CheckFn",
    "CHECKS",
    "register",
    "run_all_checks",
]

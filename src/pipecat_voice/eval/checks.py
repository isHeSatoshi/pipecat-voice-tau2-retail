"""Concrete failure-mode checks.

Three checks live here. Each is a small function that walks
``SimulationRun.messages`` looking for a specific pattern. The names
and signatures are stable; add new failure modes by writing a new file
in this package and registering it in ``pipecat_voice.eval.__init__``.

All checks are pure (no I/O, no LLM, no global state) so they can be
called from the runner, from a CLI subcommand, or from the viewer.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_messages(sim_run: SimulationRun) -> list[AssistantMessage]:
    return [m for m in (sim_run.messages or []) if isinstance(m, AssistantMessage)]


def _tool_call_signatures(sim_run: SimulationRun) -> list[tuple[str, tuple]]:
    """Return ``[(tool_name, frozenset(args.items())), ...]`` for assistant
    tool calls, in chronological order.
    """
    out: list[tuple[str, tuple]] = []
    for m in _assistant_messages(sim_run):
        if not m.tool_calls:
            continue
        for tc in m.tool_calls:
            args = tuple(sorted((tc.arguments or {}).items()))
            out.append((tc.name, args))
    return out


def _tool_results(sim_run: SimulationRun) -> list[ToolMessage]:
    return [m for m in (sim_run.messages or []) if isinstance(m, ToolMessage)]


# ---------------------------------------------------------------------------
# auth_loop
# ---------------------------------------------------------------------------


def check_auth_loop(sim_run: SimulationRun, task: Task) -> CheckResult:
    """Fail when the agent re-calls the same authentication tool with
    identical arguments within the first N turns.

    Concretely: if the same ``(tool_name, args)`` signature appears
    >= 3 times in the agent's tool-call sequence, the agent is stuck
    asking the same authentication question instead of advancing.
    """
    sigs = _tool_call_signatures(sim_run)
    if not sigs:
        return CheckResult(
            "auth_loop",
            passed=True,
            message="no tool calls",
        )
    counts = Counter(sigs)
    most_common, hits = counts.most_common(1)[0]
    if hits >= 3:
        name, args = most_common
        return CheckResult(
            "auth_loop",
            passed=False,
            message=(
                f"tool `{name}` called {hits}x with the same arguments "
                f"({dict(args)}) — agent is looping on auth"
            ),
        )
    # Also flag: any repeated (>=2) auth-style call within the first 5
    # tool calls.
    if len(sigs) >= 2:
        first_window = sigs[:5]
        first_counts = Counter(first_window)
        sig2, hits2 = first_counts.most_common(1)[0]
        if hits2 >= 2 and sig2[0].startswith("find_user_id_"):
            name, args = sig2
            return CheckResult(
                "auth_loop",
                passed=False,
                message=(
                    f"auth tool `{name}` called {hits2}x in first 5 turns "
                    f"with the same args ({dict(args)}) — agent looping on auth"
                ),
            )
    return CheckResult(
        "auth_loop",
        passed=True,
        message=f"no auth loop ({len(sigs)} unique tool-call signatures)",
    )


# ---------------------------------------------------------------------------
# no_tool_calls
# ---------------------------------------------------------------------------


def check_no_tool_calls(sim_run: SimulationRun, task: Task) -> CheckResult:
    """Fail when the agent never made a tool call.

    Pure text replies without ever inspecting state usually mean the
    agent gave up (or hallucinated an answer).
    """
    sigs = _tool_call_signatures(sim_run)
    if not sigs:
        return CheckResult(
            "no_tool_calls",
            passed=False,
            message="agent never invoked a tau2 tool",
        )
    return CheckResult(
        "no_tool_calls",
        passed=True,
        message=f"{len(sigs)} tool calls",
    )


# ---------------------------------------------------------------------------
# premature_stop
# ---------------------------------------------------------------------------


def check_premature_stop(sim_run: SimulationRun, task: Task) -> CheckResult:
    """Fail when the conversation ended with ``###STOP###`` / transfer
    while a gold action still needs to be taken.

    Concretely: if ``termination_reason`` is ``user_stop`` AND the gold
    trajectory has any write-tool action (``cancel_pending_order``,
    ``modify_pending_order_address``, ``modify_pending_order_items``,
    ``modify_pending_order_payment``, ``return_delivered_order_items``,
    ``exchange_delivered_order_items``), the conversation ended too
    early. We don't penalise user_stop for pure read-only tasks.
    """
    reason = sim_run.termination_reason
    if reason != "user_stop":
        return CheckResult(
            "premature_stop",
            passed=True,
            message=f"termination={reason!r} (not user_stop)",
        )

    write_actions = {"transfer_to_human_agents"}  # transfer is the user's escape; not a failure
    if task.evaluation_criteria is None or task.evaluation_criteria.actions is None:
        return CheckResult(
            "premature_stop",
            passed=True,
            message="no gold actions to compare",
        )
    for a in task.evaluation_criteria.actions:
        if a.name in write_actions:
            continue
        if a.requestor == "user":
            continue  # user-side actions are not the agent's job
        # Anything else that the gold trajectory calls, the agent
        # should have called or got close to. Flag if nothing was
        # attempted at all.
    sigs = _tool_call_signatures(sim_run)
    if not sigs:
        return CheckResult(
            "premature_stop",
            passed=False,
            message="user stopped with zero agent tool calls — bailed before any work",
        )
    return CheckResult(
        "premature_stop",
        passed=True,
        message=f"user_stop after {len(sigs)} tool calls",
    )

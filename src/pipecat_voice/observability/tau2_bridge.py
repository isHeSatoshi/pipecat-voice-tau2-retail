"""Convert Pipecat LLM context / frame events into a tau2 ``SimulationRun``.

The runner captures every conversation turn into the LLMContext. After the
conversation ends (stop signal, timeout, max turns), we serialise that
context into a list of tau2 ``Message`` objects so the standard tau2
evaluator (``tau2.evaluator.evaluate_simulation``) can score it.

We deliberately do NOT run tau2's turn-based orchestrator at runtime. The
runtime is Pipecat; tau2's role is the eval ground-truth. After the
conversation ends, we synthesise a ``SimulationRun`` from the captured
state and call the evaluator once.

Translation rules
----------------

For each entry in the LLMContext's ``messages`` list:

- ``system`` messages: dropped (the system prompt is part of the policy
  snapshot, not the trajectory).
- ``user`` messages: become ``UserMessage``.
- ``assistant`` messages with ``tool_calls``: become ``AssistantMessage``
  with the tool calls preserved. We synthesise ``ToolMessage`` results
  from any subsequent ``tool`` role message so the evaluator sees a
  complete trajectory.
- ``assistant`` messages with text only: become ``AssistantMessage``.
- ``tool`` messages: become ``ToolMessage`` with ``requestor="assistant"``
  (since tau2's evaluation only walks the agent's tool calls).

Termination reason:

- ``###STOP###`` from the user side → ``TerminationReason.USER_STOP``.
- ``###TRANSFER###`` → ``TerminationReason.USER_STOP`` (and a flag in info).
- Max-time / max-turns hit → ``TerminationReason.TIMEOUT``.
- Other → ``TerminationReason.AGENT_STOP``.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task

from pipecat_voice.observability.trace_writer import TraceWriter


def _next_id(prefix: str = "msg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def messages_from_context(context_messages: list[dict[str, Any]]) -> list[Message]:
    """Translate Pipecat-style OpenAI messages into tau2 ``Message`` objects.

    Filters out ``system`` messages (they are part of the policy snapshot).
    Tool calls are preserved with a synthesized ``tool_call_id``.
    """
    out: list[Message] = []
    pending_tool_ids: list[str] = []

    for m in context_messages:
        role = m.get("role")
        if role == "system":
            # System messages aren't part of tau2's evaluated trajectory.
            continue
        if role == "user":
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                # Map OpenAI tool_call shape back to tau2's ToolCall.
                from tau2.data_model.message import ToolCall  # local import for speed

                tau_tc = [
                    ToolCall(
                        id=tc.get("id", _next_id("tc")),
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"],
                        requestor="user",
                    )
                    for tc in tool_calls
                ]
                out.append(UserMessage(role="user", content=content or "", tool_calls=tau_tc))
                pending_tool_ids.extend(tc.id for tc in tau_tc)
            else:
                out.append(UserMessage(role="user", content=content))
        elif role == "assistant":
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                from tau2.data_model.message import ToolCall

                tau_tc = [
                    ToolCall(
                        id=tc.get("id", _next_id("tc")),
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"],
                        requestor="assistant",
                    )
                    for tc in tool_calls
                ]
                out.append(
                    AssistantMessage(role="assistant", content=content or None, tool_calls=tau_tc)
                )
                pending_tool_ids.extend(tc.id for tc in tau_tc)
            else:
                out.append(AssistantMessage(role="assistant", content=content))
        elif role == "tool":
            tool_call_id = m.get("tool_call_id", _next_id("tc"))
            content = m.get("content") or ""
            # Try to parse JSON, fall back to raw string.
            try:
                parsed = json.loads(content)
                content_str = json.dumps(parsed)
            except (TypeError, ValueError):
                content_str = str(content)
            out.append(
                ToolMessage(
                    id=tool_call_id,
                    role="tool",
                    content=content_str,
                    requestor="assistant",
                )
            )
            if tool_call_id in pending_tool_ids:
                pending_tool_ids.remove(tool_call_id)
        else:
            logger.warning(f"Unknown message role in context: {role!r}")
    return out


def build_simulation_run(
    *,
    task: Task,
    context_messages: list[dict[str, Any]],
    termination_reason: TerminationReason,
    duration_seconds: float,
    agent_cost: Optional[float] = None,
    user_cost: Optional[float] = None,
    extra_info: Optional[dict[str, Any]] = None,
) -> SimulationRun:
    """Build a tau2 ``SimulationRun`` from the captured conversation."""
    messages = messages_from_context(context_messages)
    return SimulationRun(
        id=_next_id("sim"),
        task_id=task.id,
        start_time=_iso_now(offset_seconds=-duration_seconds),
        end_time=_iso_now(),
        duration=duration_seconds,
        termination_reason=termination_reason.value,
        agent_cost=agent_cost,
        user_cost=user_cost,
        messages=messages,
        seed=None,
        mode="voice",
        info=extra_info or {},
    )


def _iso_now(offset_seconds: float = 0.0) -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def score_simulation_run(
    run: SimulationRun,
    task: Task,
    *,
    evaluation_type: str = "all",
) -> Any:
    """Invoke tau2's evaluator on a built ``SimulationRun``.

    Mirrors ``tau2.runner.simulation.run_simulation``'s scoring path. We do
    NOT go through the runner because we did not use the orchestrator at
    runtime — we built a SimulationRun directly and want to score it.
    """
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    etype = EvaluationType(evaluation_type)
    domain = task.user_scenario.instructions.domain if hasattr(task.user_scenario.instructions, "domain") else "retail"
    # Fall back to the environment name if instructions don't carry domain.
    # strict_replay=False: voice transcripts garble ids/args, so replayed
    # tool outputs cosmetically differ from recorded ones; aborting on that
    # would nuke the whole score instead of grading behavior.
    return evaluate_simulation(
        simulation=run,
        task=task,
        evaluation_type=etype,
        solo_mode=False,
        domain=domain,
        strict_replay=False,
    )


def run_evaluator(simulation_run: SimulationRun, task: Task) -> Any:
    """Shorthand: default ``EvaluationType.ALL`` scoring for one task."""
    return score_simulation_run(simulation_run, task, evaluation_type="all")


def run_evaluator_local(simulation_run: SimulationRun, task: Task) -> Any:
    """Score with tau2's local evaluators only (ENV + ACTION + COMMUNICATE).

    ``EvaluationType.ALL`` additionally runs NL_ASSERTIONS whenever it is in
    the task's ``reward_basis`` — but the NL judge defaults to
    ``gpt-4.1`` (OpenAI) and crashes without ``OPENAI_API_KEY``. This harness
    only carries a MiniMax key on an Anthropic-compatible endpoint, which
    litellm cannot be routed to (its Anthropic provider honors no base-URL
    env var). So we run the three LLM-free evaluators and merge exactly like
    ``ALL`` does, minus NL. The exclusion is recorded in ``info``.
    """
    from tau2.data_model.simulation import RewardInfo
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    domain = task.user_scenario.instructions.domain if hasattr(task.user_scenario.instructions, "domain") else "retail"
    kwargs: dict[str, Any] = {
        "simulation": simulation_run,
        "task": task,
        "solo_mode": False,
        "domain": domain,
        "strict_replay": False,
    }
    env_ri = evaluate_simulation(evaluation_type=EvaluationType.ENV, **kwargs)
    act_ri = evaluate_simulation(evaluation_type=EvaluationType.ACTION, **kwargs)
    comm_ri = evaluate_simulation(evaluation_type=EvaluationType.COMMUNICATE, **kwargs)

    basis = set(task.evaluation_criteria.reward_basis)
    reward = 1.0
    breakdown: dict[str, Any] = {}
    if basis & {RewardType.DB, RewardType.ENV_ASSERTION}:
        if env_ri.reward_breakdown is not None:
            breakdown.update(env_ri.reward_breakdown)
        reward *= env_ri.reward
    if basis & {RewardType.ACTION}:
        if act_ri.reward_breakdown is not None:
            breakdown.update(act_ri.reward_breakdown)
        reward *= act_ri.reward
    if basis & {RewardType.COMMUNICATE}:
        if comm_ri.reward_breakdown is not None:
            breakdown.update(comm_ri.reward_breakdown)
        reward *= comm_ri.reward

    info = {
        "env": env_ri.info,
        "nl": None,
        "communicate": comm_ri.info,
        "action": act_ri.info,
        "note": (
            "NL_ASSERTIONS excluded: the NL judge needs OPENAI_API_KEY "
            "(defaults to gpt-4.1); this harness only has a MiniMax key."
            if basis & {RewardType.NL_ASSERTION}
            else "No NL assertions in this task's reward_basis."
        ),
    }
    return RewardInfo(
        reward=reward,
        db_check=env_ri.db_check,
        env_assertions=env_ri.env_assertions,
        action_checks=act_ri.action_checks,
        nl_assertions=None,
        communicate_checks=comm_ri.communicate_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=breakdown,
        info=info,
    )

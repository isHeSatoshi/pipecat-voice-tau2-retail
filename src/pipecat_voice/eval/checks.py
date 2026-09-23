from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task

AUTH_TOOL_PREFIXES = ("find_user_id",)
WRITE_TOOL_NAMES = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}
AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|go ahead|please proceed|proceed|confirm|do it|sounds good)\b",
    re.IGNORECASE,
)
NO_ARG_TOOL_NAMES = {"list_all_product_types"}
TOOL_ERROR_MARKERS = (
    "missing 1 required",
    "missing required",
    "unexpected keyword argument",
    "unexpected_arguments",
    "invalid_tool_arguments",
    "not found",
    "error:",
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def _assistant_messages(sim_run: SimulationRun) -> list[AssistantMessage]:
    return [m for m in (sim_run.messages or []) if isinstance(m, AssistantMessage)]


def _tool_call_signatures(sim_run: SimulationRun) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for message in _assistant_messages(sim_run):
        for call in message.tool_calls or []:
            args = call.arguments or {}
            out.append((call.name, repr(sorted(args.items()))))
    return out


def check_auth_loop(sim_run: SimulationRun, task: Task) -> CheckResult:
    auth_calls = [
        signature
        for signature in _tool_call_signatures(sim_run)
        if signature[0].startswith(AUTH_TOOL_PREFIXES)
    ]
    repeated = [
        (name, count) for name, count in Counter(auth_calls).items() if count >= 2
    ]
    if repeated:
        name, count = repeated[0]
        return CheckResult(
            "auth_loop",
            False,
            f"authentication tool `{name}` repeated the same call {count} times",
        )
    auth_prompts = sum(
        1
        for message in _assistant_messages(sim_run)
        if not message.tool_calls
        and re.search(
            r"\b(authenticate|verification|verify your|zip code|email address|first name|last name)\b",
            message.content or "",
            re.IGNORECASE,
        )
    )
    if auth_prompts >= 3:
        return CheckResult(
            "auth_loop",
            False,
            f"agent asked for authentication details {auth_prompts} times",
        )
    return CheckResult(
        "auth_loop",
        True,
        f"no repeated authentication call; auth prompts={auth_prompts}",
    )


def check_tool_argument_integrity(sim_run: SimulationRun, task: Task) -> CheckResult:
    failures: list[str] = []
    for message in sim_run.messages or []:
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls or []:
                if not call.arguments and call.name not in NO_ARG_TOOL_NAMES:
                    failures.append(f"{call.name} called with empty arguments")
        elif isinstance(message, ToolMessage):
            text = message.content or ""
            if message.error or any(
                marker in text.lower() for marker in TOOL_ERROR_MARKERS
            ):
                failures.append(f"{message.id}: {text[:180]}")
    if failures:
        return CheckResult(
            "tool_argument_integrity",
            False,
            "; ".join(failures[:5]),
        )
    calls = _tool_call_signatures(sim_run)
    return CheckResult(
        "tool_argument_integrity",
        True,
        f"{len(calls)} calls passed recorded argument/error checks",
    )


def _confirmed_before(messages, index: int) -> bool:
    previous = [messages[cursor] for cursor in range(index - 1, -1, -1)]
    previous = [
        message for message in previous if getattr(message, "role", "") != "tool"
    ]
    if len(previous) < 2:
        return False
    confirmation, proposal = previous[0], previous[1]
    return bool(
        getattr(proposal, "role", "") == "assistant"
        and not getattr(proposal, "tool_calls", None)
        and getattr(proposal, "content", None)
        and getattr(confirmation, "role", "") == "user"
        and AFFIRMATIVE.search(confirmation.content or "")
    )


def check_write_protocol(sim_run: SimulationRun, task: Task) -> CheckResult:
    messages = sim_run.messages or []
    called_names: list[str] = []
    tool_results = {
        message.id: message for message in messages if isinstance(message, ToolMessage)
    }
    failures: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AssistantMessage):
            continue
        calls = message.tool_calls or []
        called_names.extend(call.name for call in calls)
        if len(calls) > 1:
            failures.append(f"batch contained {len(calls)} tool calls")
        for call in calls:
            if call.name in WRITE_TOOL_NAMES:
                if not _confirmed_before(messages, index):
                    failures.append(
                        f"{call.name} lacked immediate explicit confirmation"
                    )
                result = tool_results.get(call.id)
                if result is None:
                    failures.append(f"{call.name} has no recorded result")
                elif result.error:
                    failures.append(f"{call.name} failed: {result.content}")
    expected_writes = {
        action.name
        for action in (
            task.evaluation_criteria.actions if task.evaluation_criteria else []
        )
        or []
        if getattr(action, "requestor", "assistant") == "assistant"
        and action.name in WRITE_TOOL_NAMES
    }
    actual_writes = {name for name in called_names if name in WRITE_TOOL_NAMES}
    if expected_writes and not actual_writes:
        failures.append("no expected write action was attempted")
    if failures:
        return CheckResult("write_protocol", False, "; ".join(failures))
    return CheckResult(
        "write_protocol",
        True,
        f"write calls={len(actual_writes)}, expected write categories={len(expected_writes)}",
    )

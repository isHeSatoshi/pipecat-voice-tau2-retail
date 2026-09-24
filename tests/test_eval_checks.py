from __future__ import annotations

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task

from pipecat_voice.eval.checks import (
    check_auth_loop,
    check_tool_argument_integrity,
    check_write_protocol,
)
from pipecat_voice.tau2.environment import load_tasks_filtered


def _task() -> Task:
    return load_tasks_filtered("retail", "base", task_ids=["1"])[0]


def _run(messages) -> SimulationRun:
    return SimulationRun(
        id="sim_test",
        task_id="1",
        start_time="2026-01-01T00:00:00+00:00",
        end_time="2026-01-01T00:01:00+00:00",
        duration=60.0,
        termination_reason=TerminationReason.AGENT_STOP,
        agent_cost=None,
        user_cost=None,
        messages=messages,
        seed=42,
        mode="voice",
    )


def test_auth_loop_detects_identical_auth_calls() -> None:
    call = ToolCall(
        id="auth-1",
        name="find_user_id_by_name_zip",
        arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
        requestor="assistant",
    )
    result = _run(
        [
            AssistantMessage(role="assistant", content=None, tool_calls=[call]),
            ToolMessage(id="auth-1", role="tool", content="{}", requestor="assistant"),
            AssistantMessage(role="assistant", content=None, tool_calls=[call]),
        ]
    )
    assert not check_auth_loop(result, _task()).passed


def test_tool_argument_integrity_detects_missing_argument_error() -> None:
    call = ToolCall(
        id="read-1",
        name="get_order_details",
        arguments={},
        requestor="assistant",
    )
    result = _run(
        [
            AssistantMessage(role="assistant", content=None, tool_calls=[call]),
            ToolMessage(
                id="read-1",
                role="tool",
                content="missing 1 required positional argument: 'order_id'",
                requestor="assistant",
                error=True,
            ),
        ]
    )
    assert not check_tool_argument_integrity(result, _task()).passed


def test_write_protocol_requires_confirmation_and_expected_action() -> None:
    call = ToolCall(
        id="write-1",
        name="exchange_delivered_order_items",
        arguments={},
        requestor="assistant",
    )
    result = _run(
        [
            UserMessage(role="user", content="I want the thermostat instead."),
            AssistantMessage(role="assistant", content="I can exchange it."),
            AssistantMessage(role="assistant", content=None, tool_calls=[call]),
        ]
    )
    assert not check_write_protocol(result, _task()).passed


def test_write_protocol_accepts_confirmed_successful_write() -> None:
    call = ToolCall(
        id="write-2",
        name="exchange_delivered_order_items",
        arguments={},
        requestor="assistant",
    )
    result = _run(
        [
            AssistantMessage(
                role="assistant",
                content="I will exchange the thermostat on order #W2378156.",
            ),
            UserMessage(role="user", content="Yes, please proceed."),
            AssistantMessage(role="assistant", content=None, tool_calls=[call]),
            ToolMessage(
                id="write-2",
                role="tool",
                content='{"success": true}',
                requestor="assistant",
            ),
        ]
    )
    assert check_write_protocol(result, _task()).passed

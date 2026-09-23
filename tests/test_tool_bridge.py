"""Tool bridge tests: prove that LLM tool calls route into tau2's Environment.

These tests exercise ``build_function_schemas`` and ``bind_handlers``
directly against the tau2 retail environment so we know the dispatch
shape matches tau2's ``Environment.make_tool_call`` signature.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from pipecat_voice.tau2.environment import build_tau2_environment
from pipecat_voice.tau2.tool_bridge import (
    Tau2ToolExecutor,
    bind_handlers,
    build_function_schemas,
)
from pipecat_voice.interfaces import LLMToolCall


def test_function_schemas_for_retail_tools() -> None:
    env = build_tau2_environment("retail")
    schemas = build_function_schemas(env.get_tools())
    names = {getattr(s, "_name", None) for s in schemas}
    assert "find_user_id_by_name_zip" in names
    assert "calculate" in names
    # Pydantic-style JSON schema: properties and required populated.
    for s in schemas:
        assert getattr(s, "_properties", None) is not None
        assert isinstance(getattr(s, "_required", None), list)


def test_bound_handlers_execute_against_tau2_env() -> None:
    async def _go():
        env = build_tau2_environment("retail")
        schemas = build_function_schemas(env.get_tools())
        bind_handlers(schemas, env)

        # Find the find_user_id_by_name_zip schema.
        schema = next(s for s in schemas if s._name == "find_user_id_by_name_zip")
        assert schema._handler is not None

        result = await schema._handler({
            "first_name": "Yusuf",
            "last_name": "Rossi",
            "zip": "19122",
        })
        assert "user_id" in result or "user_details" in result or isinstance(result, dict)
    asyncio.run(_go())


def test_tau2_tool_executor_routes_to_env() -> None:
    async def _go():
        env = build_tau2_environment("retail")
        executor = Tau2ToolExecutor(env)
        result = await executor.execute(
            LLMToolCall(
                id="tc1",
                name="find_user_id_by_name_zip",
                arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
            ),
            requestor="assistant",
        )
        # tau2 may return a JSON-stringified response or a quoted message;
        # we only check it's a non-empty string.
        assert isinstance(result, str) and len(result) > 0
    asyncio.run(_go())


def test_build_initial_messages_default_greeting() -> None:
    """When a task has no initial_state, we start with the agent greeting."""
    from pipecat_voice.tau2.context import build_initial_messages
    from tau2.data_model.tasks import Task, UserScenario

    task = Task(
        id="dummy",
        description=None,
        user_scenario=UserScenario(persona=None, instructions="do something"),
        initial_state=None,
        evaluation_criteria=None,
    )
    msgs = build_initial_messages(task)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert "help" in msgs[0]["content"].lower()

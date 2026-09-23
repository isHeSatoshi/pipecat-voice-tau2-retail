"""Tau2ToolBridge — routes Pipecat LLM tool calls into tau2's Environment.

Pipecat's LLM services emit ``FunctionCallsStartedFrame`` when the LLM
decides to call a tool. We wrap each tau2 ``Tool`` into a Pipecat
``FunctionSchema`` whose ``handler`` is an async function that calls
``Environment.make_tool_call(name, requestor="assistant", **args)``.

Why wrap instead of intercept?
------------------------------

The ``handler`` callback approach keeps tool execution inside Pipecat's
standard ``LLMService.run_function_calls`` flow, which:

- Runs in parallel with the next LLM token generation (Pipecat's
  ``run_in_parallel=True`` default for ``LLMService``).
- Sets ``FunctionCallInProgressFrame`` so downstream observers can see
  per-call progress.
- Streams results back into the LLM context via
  ``FunctionCallResultFrame`` so the LLM gets tool results on the next turn.

Concretely: tau2's tool execution is synchronous and may be slow for DB
mutations, so we run the handler in a thread to avoid blocking the event
loop. The Pipecat service itself is async, but tau2's ``Environment`` is a
plain object — we hand it off via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

from loguru import logger

from tau2.environment.environment import Environment
from tau2.environment.tool import Tool


def build_function_schemas(tools: list[Tool]) -> list[Any]:
    """Convert tau2 ``Tool`` objects into Pipecat ``FunctionSchema`` instances.

    Each tau2 tool has a ``name`` and a Pydantic ``args_schema`` whose JSON
    schema we pass through to ``FunctionSchema``. The handler invokes tau2's
    ``Environment.make_tool_call`` and returns the JSON-serialized result.

    The handler is closure-bound to the environment instance so that the
    Pipecat LLM service can dispatch without knowing about tau2 internals.
    """
    try:
        from pipecat.adapters.schemas.function_schema import FunctionSchema  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Pipecat's FunctionSchema is unavailable; install pipecat-ai>=0.0.84."
        ) from e

    schemas: list[FunctionSchema] = []

    def _make_handler(env: Environment, tool_name: str):
        async def _handler(params) -> dict[str, Any]:
            """Async handler invoked by Pipecat when the LLM calls this tool."""
            arguments = _coerce_arguments(params)
            logger.debug(f"[tau2 tool] {tool_name}({arguments})")
            loop = asyncio.get_event_loop()
            # tau2 is synchronous; run in a thread to avoid blocking the loop.
            try:
                result_str = await loop.run_in_executor(
                    None,
                    _sync_make_tool_call,
                    env,
                    tool_name,
                    arguments,
                )
            except Exception as e:
                logger.warning(f"[tau2 tool] {tool_name} failed: {e}")
                result_str = json.dumps({"error": str(e)})
            # result_str is already JSON. Deliver via result_callback: Pipecat
            # 1.x writes an "IN_PROGRESS" placeholder tool message when the
            # call starts and only replaces it when the callback fires. The
            # return value alone is discarded — without the callback the LLM
            # sees IN_PROGRESS forever and re-calls the tool in a loop.
            try:
                result = json.loads(result_str)
            except (TypeError, json.JSONDecodeError):
                result = {"result": result_str}
            await _deliver_result(params, result)
            return result

        return _handler

    for tool in tools:
        # Each tau2 Tool exposes a Pydantic args_schema; convert via
        # model_json_schema() to get the standard JSON schema Pipecat expects.
        try:
            args_schema = tool.args_schema.model_json_schema()
        except Exception:
            args_schema = {"type": "object", "properties": {}}
        properties = args_schema.get("properties", {})
        required = args_schema.get("required", [])
        # Strip Pydantic-specific fields Pipecat doesn't want.
        properties = {
            name: {k: v for k, v in prop.items() if k in {"type", "description", "enum", "items"}}
            for name, prop in properties.items()
        }

        # Build the schema with a closure-captured handler. We bind env later.
        schema = FunctionSchema(
            name=tool.name,
            description=tool.short_desc or tool.name,
            properties=properties,
            required=required,
            handler=None,  # set after construction via setattr below
        )
        # Bind the handler with the captured env reference. We use a sentinel
        # attr; the runner binds it before constructing the LLM service.
        schemas.append(schema)
    return schemas


def bind_handlers(
    schemas: list[Any],
    env: Environment,
) -> list[Any]:
    """Bind each schema's ``handler`` to a function that calls ``env``.

    Idempotent: re-binding is a no-op if the handler is already set.
    """
    try:
        from pipecat.adapters.schemas.function_schema import FunctionSchema  # type: ignore
    except ImportError:
        return schemas

    for schema in schemas:
        if schema._handler is not None:
            continue
        tool_name = schema._name

        async def _handler(params, _env=env, _name=tool_name):
            arguments = _coerce_arguments(params)
            logger.debug(f"[tau2 tool] {_name}({arguments})")
            loop = asyncio.get_event_loop()
            try:
                result_str = await loop.run_in_executor(
                    None, _sync_make_tool_call, _env, _name, arguments
                )
            except Exception as e:
                logger.warning(f"[tau2 tool] {_name} failed: {e}")
                result_str = json.dumps({"error": str(e)})
            try:
                result = json.loads(result_str)
            except (TypeError, json.JSONDecodeError):
                result = {"result": result_str}
            await _deliver_result(params, result)
            return result

        schema._handler = _handler
    return schemas


async def _deliver_result(params, result: dict[str, Any]) -> None:
    """Broadcast the tool result via Pipecat's result_callback when present.

    No-op for plain-dict invocations (dummy/ProtocolLLM path), which use the
    return value directly.
    """
    cb = getattr(params, "result_callback", None)
    if callable(cb):
        try:
            await cb(result)
        except Exception as e:
            logger.warning(f"[tau2 tool] result_callback failed: {e}")


def _coerce_arguments(params) -> dict[str, Any]:
    """Extract the arguments mapping from a Pipecat handler invocation.

    Pipecat 1.x calls tool handlers with a ``FunctionCallParams`` object
    (``.arguments``); older code passed a plain dict. Accept both.
    """
    args = getattr(params, "arguments", params)
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}
    try:
        return dict(args)
    except Exception:
        return {}


def _sync_make_tool_call(env: Environment, name: str, arguments: dict[str, Any]) -> str:
    """Synchronous wrapper around ``Environment.make_tool_call``.

    Returns a JSON string suitable for the LLM context.
    """
    return env.to_json_str(env.make_tool_call(name, requestor="assistant", **arguments))


def attach_environment_to_schemas(schemas: list[Any], env: Environment) -> None:
    """Convenience wrapper around :func:`bind_handlers` (kept for clarity)."""
    bind_handlers(schemas, env)


# =============================================================================
# Direct synchronous executor (used in tests / scripted scenarios)
# =============================================================================


class Tau2ToolExecutor:
    """Plain (non-Pipecat) executor that runs one tool call against a tau2 env.

    Satisfies :class:`pipecat_voice.interfaces.ToolExecutor` so the runner
    can use it in tests and in the dummy smoke path.
    """

    def __init__(self, env: Environment):
        self._env = env

    async def execute(self, call, requestor: str = "assistant") -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _sync_make_tool_call,
            self._env,
            call.name,
            call.arguments,
        )

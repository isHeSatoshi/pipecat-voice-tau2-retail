"""Build a Pipecat ``LLMContext`` from a tau2 environment + task.

The Pipecat agent pipeline needs:

- A system prompt that mirrors tau2's ``LLMAgent.system_prompt``
  (instructions + domain policy).
- A list of tools matching the tau2 environment's tools, in Pipecat's
  ``FunctionSchema`` shape.
- An initial message history when the task has ``initial_state.message_history``.

This module is the only place that translates between tau2 data classes and
Pipecat data classes for the LLM context. Keeping it isolated means the rest
of the harness does not need to know about tau2 internals.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment

from pipecat_voice.tau2.tool_bridge import (
    ToolExecutionPolicy,
    bind_handlers,
    build_function_schemas,
)

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_TEMPLATE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def build_agent_system_prompt(
    domain_policy: str, override: Optional[str] = None
) -> str:
    """Construct the system prompt tau2's ``LLMAgent`` would use.

    If ``override`` is given (from ``--prompt-variant``), use it verbatim
    instead of templating the baseline instruction. The override is
    expected to already be a complete system prompt; we splice in the
    domain policy if the override doesn't include it. No ``str.format``
    is applied to overrides — variants own their own placeholders if
    they need them.
    """
    if override is not None:
        if "<policy>" in override and "</policy>" in override:
            return override
        return override + "\n\n<policy>\n" + domain_policy + "\n</policy>"
    return SYSTEM_PROMPT_TEMPLATE.format(
        agent_instruction=AGENT_INSTRUCTION, domain_policy=domain_policy
    )


def build_user_system_prompt(persona_text: str, scenario_text: str) -> str:
    """Construct a system prompt for the user simulator pipeline.

    Mirrors tau2's ``UserSimulator.system_prompt`` shape: global guidelines
    + persona + scenario. For the eval baseline we only include the scenario
    text; tau2's full global guidelines (with stop/transfer semantics) are
    applied at the integration boundary so the streaming LLM knows when to
    emit ``###STOP###``.
    """
    from tau2.user.user_simulator import get_global_user_sim_guidelines_voice

    guidelines = get_global_user_sim_guidelines_voice(use_tools=False)
    persona = persona_text.strip() or "No additional persona was provided."
    return (
        f"{guidelines}\n\n"
        f"<persona>\n{persona}\n</persona>\n\n"
        f"<scenario>\n{scenario_text}\n</scenario>\n\n"
        "VOICE EFFICIENCY: Keep every response under 40 spoken words. "
        "Reveal only the detail the agent just requested. Do not recap the "
        "scenario, repeat the agent's wording, or ask multiple confirmation "
        "questions in one turn."
    ).strip()


def build_initial_messages(task: Task) -> list[dict[str, Any]]:
    """Build the initial ``messages`` list for a Pipecat ``LLMContext``.

    Mirrors tau2's ``LLMAgent.get_init_state``: starts with a default agent
    greeting unless the task's initial_state supplies a different first
    message.
    """
    default_greeting = "Hi! How can I help you today?"
    history = []
    initial = task.initial_state
    if initial is not None and initial.message_history:
        for msg in initial.message_history:
            role = getattr(msg, "role", "")
            if role in {"system", "user", "assistant", "tool"}:
                history.append(_to_openai_message(msg))
    else:
        history.append({"role": "assistant", "content": default_greeting})
    return history


def _to_openai_message(msg) -> dict[str, Any]:
    """Convert a tau2 ``Message`` (or subclass) to OpenAI chat format."""
    role = getattr(msg, "role", "")
    content = getattr(msg, "content", "") or ""
    tool_calls = getattr(msg, "tool_calls", None)
    out: dict[str, Any] = {"role": role, "content": content}
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ]
    # Tool messages: include tool_call_id.
    if role == "tool":
        out["tool_call_id"] = getattr(msg, "id", "")
    return out


def build_tools_and_context(env: Environment, task: Task) -> dict[str, Any]:
    """Build everything needed to construct a Pipecat ``LLMContext``.

    Returns a dict with keys:
        ``system``        -- str, system prompt
        ``messages``      -- list of OpenAI-format messages
        ``tools``         -- list of Pipecat ``FunctionSchema`` (with handlers
                              bound to ``env``)

    Optional keyword arg ``system_prompt_override`` lets a prompt
    variant replace the baseline system prompt (CLI flag
    ``--prompt-variant``). The override is responsible for including
    the policy if it wants to.
    """
    override = globals().get("_override")
    system = build_agent_system_prompt(env.get_policy(), override=override)
    messages = build_initial_messages(task)
    schemas = build_function_schemas(env.get_tools())
    tool_policy = ToolExecutionPolicy()
    bind_handlers(schemas, env, policy=tool_policy)
    logger.debug(f"Built {len(schemas)} tau2 tool schemas for task {task.id}")
    return {
        "system": system,
        "messages": messages,
        "tools": schemas,
        "tool_policy": tool_policy,
    }


def set_agent_system_prompt_override(text: Optional[str]) -> None:
    """Set the override used by ``build_tools_and_context`` for one run.

    Module-level state; safe because the CLI drives one run at a time.
    Pass ``None`` to clear.
    """
    globals()["_override"] = text

"""tau2-bench integration: environment, tool bridge, LLM context, runner."""

from pipecat_voice.tau2.context import (
    build_agent_system_prompt,
    build_initial_messages,
    build_tools_and_context,
    build_user_system_prompt,
)
from pipecat_voice.tau2.environment import (
    build_tau2_environment,
    load_tasks,
    load_tasks_filtered,
)
from pipecat_voice.tau2.runner import Tau2EvalRunner
from pipecat_voice.tau2.tool_bridge import (
    Tau2ToolExecutor,
    attach_environment_to_schemas,
    bind_handlers,
    build_function_schemas,
)

__all__ = [
    # environment
    "build_tau2_environment",
    "load_tasks",
    "load_tasks_filtered",
    # tool bridge
    "Tau2ToolExecutor",
    "build_function_schemas",
    "bind_handlers",
    "attach_environment_to_schemas",
    # context
    "build_agent_system_prompt",
    "build_initial_messages",
    "build_user_system_prompt",
    "build_tools_and_context",
    # runner
    "Tau2EvalRunner",
]

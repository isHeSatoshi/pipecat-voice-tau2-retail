"""tau2 environment construction.

Loads a tau2 ``Environment`` for the configured domain (default: retail) and
exposes helpers used by the Pipecat agent pipeline:

- :func:`build_tau2_environment` — constructs a fresh ``Environment`` for one
  task.
- :func:`load_tasks` — loads the task list and (optional) split.
- :func:`initial_message_history` — replays the task's ``initial_state`` into
  the environment so DB state and message history match the spec before the
  conversation starts.
"""
from __future__ import annotations

from typing import Optional

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.registry import registry


def build_tau2_environment(domain: str = "retail") -> Environment:
    """Construct a fresh ``Environment`` for the given domain.

    Uses tau2's registry to resolve the domain name to its constructor. The
    retail domain does not support solo mode (raises ``ValueError``); we do
    not pass ``solo_mode=True`` here.
    """
    env_constructor = registry.get_env_constructor(domain)
    return env_constructor()


def load_tasks(domain: str = "retail", split: str = "base") -> list[Task]:
    """Load the task list for a domain + split via tau2's registry."""
    tasks_loader = registry.get_tasks_loader(domain)
    return tasks_loader(split)


def load_tasks_filtered(
    domain: str,
    split: str,
    task_ids: Optional[list[str]] = None,
    max_tasks: Optional[int] = None,
) -> list[Task]:
    """Load + optionally filter + truncate tasks.

    Args:
        domain: Domain name (e.g. ``"retail"``).
        split: Task split name (e.g. ``"base"``, ``"train"``, ``"test"``).
        task_ids: If non-empty, only tasks whose ``id`` is in this list are
            returned (in the same order as ``task_ids``).
        max_tasks: If non-None, truncate the result to this many tasks.
    """
    tasks = load_tasks(domain, split)
    if task_ids:
        order = {tid: i for i, tid in enumerate(task_ids)}
        tasks = [t for t in tasks if t.id in order]
        tasks.sort(key=lambda t: order[t.id])
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks

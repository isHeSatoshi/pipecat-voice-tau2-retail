"""Baseline agent system prompt — verbatim from tau2's ``LLMAgent``.

Used as the default. Any prompt variant added later (e.g. ``v1.py``) is
expected to be a small edit on this baseline, not a rewrite.

The runner expects ``AGENT_SYSTEM_PROMPT`` to be a *complete* system
prompt without ``str.format`` placeholders. The domain policy is
appended automatically by ``build_agent_system_prompt``; do not include
it here.
"""
from __future__ import annotations

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

AGENT_SYSTEM_PROMPT = (
    "<instructions>\n"
    + AGENT_INSTRUCTION
    + "\n</instructions>"
)

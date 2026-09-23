"""Prompt variant v3 — error recovery on top of v2.

v2 advanced past auth but stalled re-calling get_order_details with a
mangled id (voice STT drops the '#' and adds dashes). v3 adds: treat a
tool error as information, fix the argument once from context, retry at
most once, then ask the user — never blind-retry the same call.
"""
from __future__ import annotations

from pipecat_voice.prompts.v2 import ADVANCE_RULE, TOOL_DISCIPLINE, VOICE_TURN_RULES
from pipecat_voice.prompts.baseline import AGENT_INSTRUCTION

ERROR_RECOVERY = """
Recovering from tool errors (read carefully):
- A failed tool call is information, not a cue to retry identically.
  Read the error, fix the argument, and retry AT MOST ONCE.
- Identifiers must be used EXACTLY as given: order ids keep their '#'
  prefix and contain no added dashes or spaces (e.g. '#W2378156', not
  'W2378-156'). Copy ids from tool results or your confirmed read-back,
  never reformat them by hand.
- If the retry also fails, stop calling tools and ask the user to
  confirm the identifier (read it back character by character).
""".strip()

AGENT_SYSTEM_PROMPT = (
    "<instructions>\n"
    + AGENT_INSTRUCTION
    + "\n\n"
    + VOICE_TURN_RULES
    + "\n\n"
    + TOOL_DISCIPLINE
    + "\n\n"
    + ADVANCE_RULE
    + "\n\n"
    + ERROR_RECOVERY
    + "\n</instructions>"
)

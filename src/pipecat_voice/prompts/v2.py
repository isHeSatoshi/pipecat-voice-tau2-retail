"""Prompt variant v2 — stronger anti-loop medicine on top of v1.

v1 cut repeats but the agent still re-called the same lookup ~11x: it
did not connect "I have the user id" to "move on". v2 makes the advance
explicit and unconditional: the moment a lookup returns a user id, the
next action MUST be an order/product tool, never another lookup.
"""
from __future__ import annotations

from pipecat_voice.prompts.v1 import TOOL_DISCIPLINE, VOICE_TURN_RULES
from pipecat_voice.prompts.baseline import AGENT_INSTRUCTION

ADVANCE_RULE = """
Advancing past authentication (read carefully — this fixes the most
common failure):
- The instant find_user_id_by_name_zip returns a user id, authentication
  is DONE. Your very next action must be an order or product tool
  (get_order_details, get_product_details, get_user_details, or the
  modify_/exchange_ tool the task needs).
- Calling any find_user_id tool when a user id is already visible in
  this conversation is a bug. Do not re-verify, do not double-check.
- If the user repeats information you already used (name, zip, order
  number), acknowledge briefly and continue the task — do not restart
  authentication.
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
    + "\n</instructions>"
)

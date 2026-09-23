"""Prompt variant v1 — fixes the three baseline failure behaviors.

Targets (see pipecat_voice.eval.checks):
- auth_loop: authenticate exactly once, then advance; never repeat a call.
- no_tool_calls: always act through tools; pure-text stalls are a failure.
- premature_stop / drift: confirm heard identifiers by read-back, keep
  voice replies to 1-2 sentences so turns stay fast and transcribable.

Prompt-only change: same policy, same tools, no pipeline edits.
"""
from __future__ import annotations

from pipecat_voice.prompts.baseline import AGENT_INSTRUCTION

VOICE_TURN_RULES = """
Voice-channel rules (you are on a phone call, the user hears everything
you say and speech recognition may garble names and numbers):
- Keep every spoken reply to 1-2 short sentences.
- When the user gives a name, zip code, order id, or product detail,
  read it back once for confirmation before using it in a tool call.
""".strip()

TOOL_DISCIPLINE = """
Tool discipline (violations fail the evaluation):
- Authenticate EXACTLY ONCE per call with find_user_id_by_name_zip using
  the exact argument names first_name, last_name, zip. Save the returned
  user id and never call any find_user_id tool again in this call.
- If a lookup returns "User not found", ask the user to respell the name
  or zip ONCE, retry at most one more time, then continue with the order
  work instead of looping.
- NEVER repeat a tool call with identical arguments. Reuse results already
  in the conversation.
- After authentication, immediately call the order/product tools the task
  needs (get_order_details, get_product_details, modify_*/exchange_*).
  Do not end your turn with pure text when a tool call is owed, and do
  not ask the user for information they already gave.
""".strip()

AGENT_SYSTEM_PROMPT = (
    "<instructions>\n"
    + AGENT_INSTRUCTION
    + "\n\n"
    + VOICE_TURN_RULES
    + "\n\n"
    + TOOL_DISCIPLINE
    + "\n</instructions>"
)

"""Dump messages from task 0 to see what happened."""
import json
from pathlib import Path

p = Path("data/runs/baseline/task_0/sim_fb1d6e72/trajectory.json")
data = json.loads(p.read_text(encoding="utf-8"))
msgs = data.get("messages") or []
print(f"=== {len(msgs)} messages ===")
for i, m in enumerate(msgs):
    role = m.get("role", "?")
    content = (m.get("content") or "")[:200]
    tcs = m.get("tool_calls") or []
    print(f"[{i}] {role}: {content!r}")
    if tcs:
        for tc in tcs:
            print(f"    tool_call: {tc.get('name')}({tc.get('arguments')})")

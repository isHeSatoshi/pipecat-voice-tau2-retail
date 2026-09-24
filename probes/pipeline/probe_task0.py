"""Inspect task 0 trajectory for failure modes."""
import json
from collections import Counter
from pathlib import Path

p = Path("data/runs/baseline/task_0/sim_fb1d6e72/trajectory.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("termination:", data.get("termination_reason"))
print("duration:", round(data.get("duration", 0), 1), "s")
ri = data.get("reward_info") or {}
print("reward:", ri.get("reward"))
print("db_match:", (ri.get("db_check") or {}).get("db_match"))
acts = ri.get("action_checks") or []
print("actions:", len(acts), "matched:", sum(1 for a in acts if a.get("action_match")))
msgs = data.get("messages") or []
print("messages:", len(msgs))
roles = Counter(m.get("role") for m in msgs)
print("roles:", dict(roles))

tcsigs = []
for m in msgs:
    if m.get("role") == "assistant" and m.get("tool_calls"):
        for tc in m["tool_calls"]:
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tcsigs.append((tc.get("name"), tuple(sorted((args or {}).items()))))
sig_counts = Counter(tcsigs)
print("tool call signatures (name, count):")
for sig, n in sig_counts.most_common(8):
    print(f"  {sig[0]}: {n}x {dict(sig[1]) if sig[1] else ''}")

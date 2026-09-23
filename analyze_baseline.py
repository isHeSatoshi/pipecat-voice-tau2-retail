"""Compact baseline analysis: per-task turns, tool calls, reward state."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

for tid in ["0", "1", "2", "3", "4"]:
    paths = glob.glob(f"data/runs/baseline_m27/task_{tid}/*/trajectory.json")
    if not paths:
        print(f"task {tid}: NO TRAJECTORY")
        continue
    d = json.load(open(paths[0]))
    msgs = d.get("messages", [])
    n_user = sum(1 for m in msgs if m.get("role") == "user")
    n_asst = sum(1 for m in msgs if m.get("role") == "assistant")
    n_tool = sum(1 for m in msgs if m.get("role") == "tool")
    calls = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            calls.append(tc.get("name"))
    ri = d.get("reward_info")
    print(f"--- task {tid}: msgs={len(msgs)} (u={n_user} a={n_asst} t={n_tool}) "
          f"term={d.get('termination_reason')} reward_info={'None' if ri is None else 'present'}")
    print(f"    tool calls: {calls[:12]}{'...' if len(calls) > 12 else ''} (total {len(calls)})")
    # show last assistant text + first user text for flavor
    first_u = next((m.get("content", "")[:80] for m in msgs if m.get("role") == "user"), "")
    print(f"    first user: {first_u!r}")

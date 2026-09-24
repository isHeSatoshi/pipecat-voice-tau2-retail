"""Probe retail task 0-4 gold actions for fix planning."""
import sys; sys.path.insert(0, '.')
from pipecat_voice.tau2.environment import load_tasks_filtered

WRITES = {
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "transfer_to_human_agents",
}
for t in load_tasks_filtered("retail", "base", task_ids=["0","1","2","3","4"]):
    actions = (
        [a.name for a in (t.evaluation_criteria.actions or [])]
        if t.evaluation_criteria else []
    )
    writes = [a for a in actions if a in WRITES]
    reads = [a for a in actions if a not in WRITES]
    inst = t.user_scenario.instructions
    reason = inst.reason_for_call if hasattr(inst, "reason_for_call") else "(none)"
    print(f"task {t.id}: writes={writes} ; reads={reads}")
    print(f"          reason: {reason[:140]}")

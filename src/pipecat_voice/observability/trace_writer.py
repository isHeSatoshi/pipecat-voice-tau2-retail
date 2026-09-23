"""TraceWriter — JSONL event log + tau2 SimulationRun dump.

Every conversation emits a per-task event log (``voice_trace.jsonl``) and a
canonical tau2 ``SimulationRun`` JSON dump (``trajectory.json``). Together
these give offline analysis the data it needs to answer:

- Which tools did the agent call and with what arguments?
- Did the STT round-trip preserve the user's intended meaning?
- Where did the conversation diverge from the gold trajectory?
- What was the latency profile (STT, LLM, tool, TTS) per turn?

API
---

The writer is just a file appender: ``emit(event_dict)``. Each event is a
flat dict with at minimum a ``type`` field. The runner adds high-resolution
timestamps and per-turn ids.

After the run, ``dump_simulation_run(simulation_run)`` writes the full tau2
``SimulationRun`` (with messages, tool calls, termination reason, reward
info) to ``trajectory.json``.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO


@dataclass
class TraceWriter:
    """Per-task event log writer.

    Args:
        out_dir: Directory under which ``voice_trace.jsonl`` and
            ``trajectory.json`` are written. Created if missing.
        task_id: tau2 task id.
        simulation_id: Stable id for this simulation (defaults to uuid4).
    """

    out_dir: Path
    task_id: str
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _fp: IO[str] | None = field(default=None, init=False, repr=False)
    _t0: float = field(default_factory=time.time, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._fp = (self.out_dir / "voice_trace.jsonl").open("a", encoding="utf-8")

    @property
    def trace_path(self) -> Path:
        return self.out_dir / "voice_trace.jsonl"

    @property
    def trajectory_path(self) -> Path:
        return self.out_dir / "trajectory.json"

    def emit(self, event: dict[str, Any]) -> None:
        """Append one event to the JSONL log.

        Adds ``task_id``, ``simulation_id``, and ``t_offset_ms`` if missing.

        After ``close()``, emits become no-ops so background tasks that
        fire after the runner has finished (e.g. a stopper goroutine)
        do not raise.
        """
        if self._fp is None:
            return
        record = {
            "task_id": self.task_id,
            "simulation_id": self.simulation_id,
            "t_offset_ms": round((time.time() - self._t0) * 1000, 2),
            **event,
        }
        try:
            self._fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fp.flush()
        except Exception:
            pass

    def emit_meta(self, config_snapshot: dict[str, Any]) -> None:
        """Write a ``meta`` event with the resolved config snapshot."""
        self.emit({"type": "meta", "config": config_snapshot})

    def dump_simulation_run(self, simulation_run) -> Path:
        """Dump a tau2 ``SimulationRun`` as JSON to ``trajectory.json``.

        Returns the written path. Accepts any object with a ``model_dump()``
        method (tau2's Pydantic models) or a plain dict for testing.
        """
        if hasattr(simulation_run, "model_dump"):
            data = simulation_run.model_dump(mode="json")
        elif isinstance(simulation_run, dict):
            data = simulation_run
        else:
            data = {"repr": str(simulation_run)}
        with self.trajectory_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return self.trajectory_path

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

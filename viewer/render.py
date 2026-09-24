"""Rendering helpers for the pipecat_voice viewer.

Pure-functions-style helpers that take parsed data and return either
Streamlit fragments (via ``st.*`` calls) or plain Python data structures
suitable for plotting. The viewer file ``app.py`` orchestrates page
layout; this module just renders.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from viewer.run_loader import SimView, messages, reward_summary

# ---------------------------------------------------------------------------
# Compact cells
# ---------------------------------------------------------------------------


def _shorten(text: Any, n: int = 280) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= n else s[: n - 1] + "…"


def _bool_icon(b: Any) -> str:
    if b is True:
        return "✅"
    if b is False:
        return "❌"
    return "—"


# ---------------------------------------------------------------------------
# Run header
# ---------------------------------------------------------------------------


def render_run_header(sim: SimView) -> None:
    traj = sim.trajectory or {}
    rs = reward_summary(sim)
    cols = st.columns(6)
    cols[0].metric("Task", sim.task_id)
    if rs["strict_reward_available"]:
        cols[1].metric(
            "Reward", f"{rs['reward']:.3f}" if rs["reward"] is not None else "—"
        )
    else:
        partial = rs["partial_reward"]
        cols[1].metric("Local reward", f"{partial:.3f}" if partial is not None else "—")
    cols[2].metric(
        "DB match",
        _bool_icon(rs["db_match"]),
    )
    cols[3].metric(
        "Actions matched",
        f"{rs['actions_matched']}/{rs['actions_total']}",
    )
    cols[4].metric("Termination", traj.get("termination_reason", "—") or "—")
    seed = traj.get("seed", getattr(sim, "seed", None))
    cols[5].metric("Seed", "—" if seed is None else str(seed))
    sub_cols = st.columns(3)
    sub_cols[0].metric("Duration (s)", f"{traj.get('duration', 0):.1f}")
    sub_cols[1].metric(
        "Env assertions", f"{rs['env_assertions_passed']}/{rs['env_assertions_total']}"
    )
    sub_cols[2].metric("Sim id", sim.sim_id)
    if not rs["strict_reward_available"]:
        st.caption(
            "Local reward uses DB, ACTION, and COMMUNICATE. Strict Tau2 reward is unavailable because NL assertions are not scored."
        )


# ---------------------------------------------------------------------------
# Transcript view
# ---------------------------------------------------------------------------


def _tool_calls_summary(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    parts = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name") or (tc.get("function") or {}).get("name") or "?"
        args = tc.get("arguments")
        if args is None:
            args = (tc.get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        parts.append(
            f"`{name}`({_shorten(json.dumps(args, ensure_ascii=False) if args is not None else '', n=120)})"
        )
    return " · ".join(parts)


def render_transcript(sim: SimView) -> None:
    msgs = messages(sim)
    if not msgs:
        st.info("No messages in trajectory.")
        return

    if sim.audio_manifest.get("pairing_status") == "legacy_unverified":
        st.warning(
            "This saved run predates exact per-turn audio markers. Per-message audio is disabled to avoid attaching the wrong speaker audio; the original files are preserved."
        )

    if sim.conversation_audio is not None:
        with st.expander("optional mixed conversation reference"):
            st.caption("This track intentionally mixes both voices; use each message audio below for exact speaker-specific playback.")
            st.audio(str(sim.conversation_audio))

    # Group consecutive tool + tool-result pairs so the reader sees the
    # call and its response side by side.
    pending_tool_ids: set[str] = set()
    grouped: list[tuple[str, dict[str, Any]]] = []
    tool_results: dict[str, dict[str, Any]] = {}
    for m in msgs:
        role = m.get("role")
        tcs = m.get("tool_calls") or []
        if role == "assistant" and tcs:
            for tc in tcs:
                tid = tc.get("id")
                if tid:
                    pending_tool_ids.add(tid)
        if role == "tool":
            tid = m.get("tool_call_id")
            if tid is not None:
                tool_results[tid] = m
        grouped.append((role or "?", m))

    for idx, (role, m) in enumerate(grouped):
        content = m.get("content") or ""
        tcs = m.get("tool_calls") or []
        audio = m.get("audio_path")

        if role == "user":
            with st.chat_message("user", avatar="🙋"):
                st.markdown(_shorten(content, 4000))
                if audio:
                    st.audio(audio)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                if content:
                    st.markdown(_shorten(content, 4000))
                if tcs:
                    for tc in tcs:
                        tid = tc.get("id")
                        args = tc.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = None
                        st.markdown(f"🔧 **tool call** `{tc.get('name', '?')}`")
                        if args is not None:
                            st.json(args)
                        # Pair the call with its tool result, if any.
                        if tid and tid in tool_results:
                            tr = tool_results[tid]
                            tr_content = tr.get("content") or ""
                            tr_err = tr.get("error", False)
                            label = (
                                "tool result" if not tr_err else "tool result (error)"
                            )
                            st.markdown(f"↪ **{label}**")
                            try:
                                parsed = json.loads(tr_content)
                                st.json(parsed)
                            except (TypeError, json.JSONDecodeError):
                                st.code(_shorten(tr_content, 4000))
                        elif tid in pending_tool_ids:
                            st.caption("(no tool result recorded)")
                if audio:
                    st.audio(audio)
        elif role == "tool":
            # Already shown next to the call.
            pass
        else:
            with st.chat_message("system", avatar="ℹ️"):
                st.markdown(_shorten(content, 4000))


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------


def render_reward(sim: SimView) -> None:
    traj = sim.trajectory or {}
    ri = traj.get("reward_info") or {}
    if not ri:
        st.info("No reward_info on this simulation.")
        return

    info = ri.get("info") or {}
    strict_available = info.get("strict_reward_available", True)
    if strict_available:
        st.subheader(f"Final reward: **{ri.get('reward', '—')}**")
    else:
        st.subheader(f"Local reward: **{info.get('partial_reward', '—')}**")
        st.warning(
            "Strict Tau2 reward is unavailable because this task requires an NL assertion and no compatible judge is configured."
        )
    basis = ri.get("reward_basis") or []
    if basis:
        st.caption("Reward basis: " + ", ".join(str(b) for b in basis))

    # DB check.
    db = ri.get("db_check")
    if db:
        with st.expander("DB check", expanded=True):
            cols = st.columns(2)
            cols[0].metric("db_match", _bool_icon(db.get("db_match")))
            cols[1].metric("db_reward", f"{db.get('db_reward', '—')}")

    # Action checks.
    actions = ri.get("action_checks") or []
    if actions:
        st.markdown(f"**Action checks** ({len(actions)})")
        rows = []
        for a in actions:
            act = a.get("action") or {}
            rows.append(
                {
                    "match": _bool_icon(a.get("action_match")),
                    "name": act.get("name"),
                    "args": json.dumps(act.get("arguments") or {}, ensure_ascii=False),
                    "reward": a.get("action_reward"),
                    "type": (a.get("tool_type") or "—"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

    # Env assertions.
    env_a = ri.get("env_assertions") or []
    if env_a:
        st.markdown(f"**Env assertions** ({len(env_a)})")
        rows = [
            {
                "met": _bool_icon(a.get("met")),
                "reward": a.get("reward"),
                "func": (a.get("env_assertion") or {}).get("func_name"),
                "message": (a.get("env_assertion") or {}).get("message"),
            }
            for a in env_a
        ]
        st.dataframe(rows, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Trace timeline
# ---------------------------------------------------------------------------


EVENT_COLOURS = {
    "meta": "#888",
    "run_start": "#1f77b4",
    "run_end": "#2ca02c",
    "error": "#d62728",
    "stop_signal": "#ff7f0e",
    "llm_call": "#9467bd",
    "tool_call": "#8c564b",
    "tool_result": "#17becf",
    "stt_result": "#e377c2",
    "vad_start": "#f7b6d2",
    "vad_stop": "#c51b8a",
    "llm_start": "#cab2d6",
    "llm_end": "#8856a7",
    "llm_run": "#9467bd",
    "tts_start": "#c7c7c7",
    "tts_stop": "#7f7f7f",
    "frame_error": "#d62728",
}


def _event_colour(t: str) -> str:
    return EVENT_COLOURS.get(t, "#bcbd22")


def render_trace_timeline(sim: SimView) -> None:
    events = sim.trace_events or []
    if not events:
        st.info("No trace events recorded.")
        return

    # Build a list of (t_offset_ms, type, summary).
    rows = []
    for ev in events:
        t_ms = ev.get("t_offset_ms", 0)
        et = ev.get("type", "?")
        rows.append((t_ms, et, ev))

    rows.sort(key=lambda r: r[0])

    # Show a horizontal bar chart of events on a timeline (matplotlib).
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.0))
    types_in_order: list[str] = []
    for t_ms, et, _ in rows:
        if et not in types_in_order:
            types_in_order.append(et)

    for t_ms, et, ev in rows:
        y = types_in_order.index(et)
        ax.scatter(t_ms / 1000.0, y, color=_event_colour(et), s=70, zorder=3)

    ax.set_yticks(range(len(types_in_order)))
    ax.set_yticklabels(types_in_order)
    ax.set_xlabel("t (seconds from run start)")
    ax.set_title(f"Trace timeline — {len(events)} events")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Show the full event list in a dataframe.
    summary_rows = [
        {
            "t (s)": round(t_ms / 1000.0, 2),
            "type": et,
            "summary": _shorten(
                json.dumps(
                    {
                        k: v
                        for k, v in ev.items()
                        if k not in {"task_id", "simulation_id", "t_offset_ms"}
                    },
                    default=str,
                ),
                200,
            ),
        }
        for t_ms, et, ev in rows
    ]
    st.dataframe(summary_rows, width="stretch", hide_index=True)

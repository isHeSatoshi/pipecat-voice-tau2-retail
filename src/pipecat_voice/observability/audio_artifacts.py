from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _read_wave(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        data = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit PCM audio: {path}")
    samples = np.frombuffer(data, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype("<i2")
    return samples, sample_rate, channels


def _write_wave(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _turns_from_trace(trace_path: Path) -> dict[str, list[dict[str, float]]]:
    events: dict[tuple, dict[str, Any]] = {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") not in {"tts_start", "tts_stop"}:
            continue
        key = (
            event.get("type"),
            event.get("side"),
            round(float(event.get("t_offset_ms", 0)), 2),
        )
        events[key] = event
    turns: dict[str, list[dict[str, float]]] = {"agent": [], "user": []}
    open_turns: dict[str, float] = {}
    for event in sorted(events.values(), key=lambda item: item.get("t_offset_ms", 0)):
        side = event.get("side")
        if side not in turns:
            continue
        offset = float(event.get("t_offset_ms", 0))
        if event.get("type") == "tts_start":
            open_turns.setdefault(side, offset)
        elif side in open_turns:
            turns[side].append({"start_ms": open_turns.pop(side), "stop_ms": offset})
    return turns


def _silence_splits(samples: np.ndarray, sample_rate: int, count: int) -> list[int]:
    if count <= 1 or samples.size == 0:
        return []
    frame_size = max(1, int(sample_rate * 0.02))
    frame_count = samples.size // frame_size
    if frame_count < count:
        return []
    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    threshold = max(120.0, float(np.percentile(rms, 25) * 2.5))
    quiet = rms < threshold
    candidates: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.append(quiet, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= max(5, int(0.20 * sample_rate / frame_size)):
                candidates.append((start * frame_size, index * frame_size))
            start = None
    if len(candidates) < count - 1:
        weights = np.linspace(0, 1, count + 1)[1:-1]
        return [int(samples.size * weight) for weight in weights]
    selected = sorted(candidates, key=lambda item: item[1] - item[0], reverse=True)[
        : count - 1
    ]
    return sorted((start + stop) // 2 for start, stop in selected)


def _message_indices(messages: list[dict[str, Any]], side: str) -> list[int]:
    indices = []
    for index, message in enumerate(messages):
        if message.get("role") != side or not message.get("content"):
            continue
        if (
            side == "assistant"
            and index == 0
            and str(message["content"]).startswith("Hi! How can I help you today?")
        ):
            continue
        indices.append(index)
    return indices


def _turn_weight_splits(
    samples: np.ndarray, turns: list[dict[str, float]]
) -> list[int]:
    if len(turns) <= 1 or samples.size == 0:
        return []
    weights = np.asarray(
        [max(1.0, turn["stop_ms"] - turn["start_ms"]) for turn in turns],
        dtype=np.float64,
    )
    boundaries = np.cumsum(weights / weights.sum() * samples.size).astype(np.int64)
    return sorted(set(boundaries.tolist()))


def build_artifacts(sim_dir: Path) -> dict[str, Any]:
    trajectory_path = sim_dir / "trajectory.json"
    trace_path = sim_dir / "voice_trace.jsonl"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    messages = trajectory.get("messages") or []
    turns = _turns_from_trace(trace_path)
    segments: list[dict[str, Any]] = []
    side_samples: dict[str, tuple[np.ndarray, int]] = {}
    for side in ("agent", "user"):
        source_name = "agent_audio.wav" if side == "agent" else "user_audio.wav"
        samples, sample_rate, _ = _read_wave(sim_dir / source_name)
        side_samples[side] = (samples, sample_rate)
        boundaries = [0, *_turn_weight_splits(samples, turns[side]), samples.size]
        role = "assistant" if side == "agent" else "user"
        message_indices = _message_indices(messages, role)
        for index, turn in enumerate(turns[side]):
            start = boundaries[index]
            stop = boundaries[index + 1]
            clip = samples[start:stop]
            path = sim_dir / "audio" / f"{side}_turn_{index + 1:02d}.wav"
            _write_wave(path, clip, sample_rate)
            segments.append(
                {
                    "side": side,
                    "index": index + 1,
                    "path": str(path.relative_to(sim_dir)),
                    "start_ms": turn["start_ms"],
                    "duration_ms": round(clip.size / sample_rate * 1000, 2),
                    "message_index": message_indices[index]
                    if index < len(message_indices)
                    else None,
                }
            )
    sample_rate = side_samples["agent"][1]
    max_ms = (
        max(
            (segment["start_ms"] + segment["duration_ms"] for segment in segments),
            default=0.0,
        )
        + 1000.0
    )
    mixed = np.zeros(int(max_ms * sample_rate / 1000) + sample_rate, dtype=np.float32)
    counts = np.zeros(mixed.size, dtype=np.int16)
    for segment in segments:
        samples, _, _ = _read_wave(sim_dir / segment["path"])
        offset = int(segment["start_ms"] * sample_rate / 1000)
        end = min(mixed.size, offset + samples.size)
        length = max(0, end - offset)
        if length:
            mixed[offset:end] += samples[:length].astype(np.float32)
            counts[offset:end] += 1
    conversation = np.divide(
        mixed,
        np.maximum(counts, 1),
        out=np.zeros_like(mixed),
        where=counts > 0,
    )
    conversation = np.clip(conversation, -32768, 32767).astype("<i2")
    conversation_path = sim_dir / "conversation.wav"
    _write_wave(conversation_path, conversation, sample_rate)
    manifest = {
        "conversation": "conversation.wav",
        "segments": segments,
    }
    (sim_dir / "audio_segments.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_dir", type=Path)
    args = parser.parse_args()
    manifest = build_artifacts(args.sim_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

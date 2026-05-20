#!/usr/bin/env python3
"""Replay collected selector JSON files through the runtime channel selector."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from channel_selector_interfaces import (
    ChannelSelectorInput,
    FrameFeatureSnapshot,
    StateSnapshot,
)
from src.runtime_selector import ACTION_TO_CHANNEL, RuntimeChannelSelector


COMPUTE_COST = {
    "kalman": 0.05,
    "gmc": 0.15,
    "inference": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the runtime channel selector on collected JSON.")
    parser.add_argument("--data", nargs="+", default=["data/labels/*.json"], help="JSON files or glob patterns.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Selector checkpoint path.")
    parser.add_argument("--output", default="results/runtime_selector_eval.json", help="Output summary JSON.")
    parser.add_argument("--low", type=float, default=0.3, help="Old frame-diff low threshold.")
    parser.add_argument("--high", type=float, default=0.7, help="Old frame-diff high threshold.")
    parser.add_argument("--max-skip-frames", type=int, default=60)
    parser.add_argument("--force-gpu-interval", type=int, default=60)
    parser.add_argument("--uncertainty-threshold", type=float, default=100000.0)
    parser.add_argument("--prediction-error-threshold", type=float, default=0.9)
    return parser.parse_args()


def expand_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matched = sorted(Path().glob(pattern))
        if matched:
            paths.extend(path for path in matched if path.is_file())
        else:
            path = Path(pattern)
            if path.is_file():
                paths.append(path)
    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise FileNotFoundError(f"No JSON files matched: {list(patterns)}")
    return unique_paths


def frame_from_features(frame_id: int, features: Dict[str, Any]) -> FrameFeatureSnapshot:
    return FrameFeatureSnapshot(
        frame_id=frame_id,
        frame_diff_mean=float(features.get("frame_diff_mean", 0.0) or 0.0),
        frame_diff_std=float(features.get("frame_diff_std", 0.0) or 0.0),
        global_motion_dx=float(features.get("global_motion_dx", 0.0) or 0.0),
        global_motion_dy=float(features.get("global_motion_dy", 0.0) or 0.0),
        optical_flow_valid_ratio=float(features.get("optical_flow_valid_ratio", 0.0) or 0.0),
        optical_flow_residual=float(features.get("optical_flow_residual", 0.0) or 0.0),
        tracked_point_count=int(features.get("tracked_point_count", 0) or 0),
        is_bootstrap_frame=bool(features.get("is_bootstrap_frame", False)),
    )


def state_from_features(frame_id: int, features: Dict[str, Any]) -> StateSnapshot:
    frames_since_gpu = int(features.get("frames_since_last_gpu", -1) or -1)
    last_gpu_frame_id = frame_id - frames_since_gpu if frames_since_gpu >= 0 else -1
    return StateSnapshot(
        frame_id=frame_id,
        last_gpu_frame_id=last_gpu_frame_id,
        last_gpu_box_count=int(features.get("last_gpu_box_count", 0) or 0),
        last_gpu_max_conf=float(features.get("last_gpu_max_conf", 0.0) or 0.0),
        recent_gpu_box_count_mean=float(features.get("recent_gpu_box_count_mean", 0.0) or 0.0),
        recent_gpu_box_count_delta=float(features.get("recent_gpu_box_count_delta", 0.0) or 0.0),
        tracker_count=int(features.get("tracker_count", 0) or 0),
        confirmed_tracker_count=int(features.get("confirmed_tracker_count", 0) or 0),
        mean_track_age=float(features.get("mean_track_age", 0.0) or 0.0),
        mean_time_since_update=float(features.get("mean_time_since_update", 0.0) or 0.0),
        max_time_since_update=int(features.get("max_time_since_update", 0) or 0),
        mean_speed=float(features.get("mean_speed", 0.0) or 0.0),
        mean_position_uncertainty=float(features.get("mean_position_uncertainty", 0.0) or 0.0),
        max_position_uncertainty=float(features.get("max_position_uncertainty", 0.0) or 0.0),
        prediction_error_ma=float(features.get("prediction_error_ma", 0.0) or 0.0),
        prediction_error_p95=float(features.get("prediction_error_p95", 0.0) or 0.0),
    )


def channel_f1(frame_info: Dict[str, Any], channel: str) -> float:
    channel_info = frame_info.get("channels", {}).get(channel, {})
    metrics = channel_info.get("metrics", {}) if isinstance(channel_info, dict) else {}
    return float(metrics.get("f1", 0.0) or 0.0)


def fixed_threshold_channel(features: Dict[str, Any], low: float, high: float) -> str:
    score = float(features.get("frame_diff_mean", 0.0) or 0.0)
    if bool(features.get("is_bootstrap_frame", False)):
        return "inference"
    if score < low:
        return "kalman"
    if score < high:
        return "gmc"
    return "inference"


def summarize(name: str, channels: List[str], f1_values: List[float]) -> Dict[str, Any]:
    counts = Counter(channels)
    total = max(1, len(channels))
    avg_compute = float(np.mean([COMPUTE_COST[channel] for channel in channels])) if channels else 0.0
    avg_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    return {
        "name": name,
        "frames": len(channels),
        "avg_f1": avg_f1,
        "avg_compute": avg_compute,
        "gpu_ratio": counts["inference"] / total,
        "channel_counts": dict(counts),
    }


def evaluate_file(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    selector = RuntimeChannelSelector(
        checkpoint_path=args.checkpoint,
        fallback_low=args.low,
        fallback_high=args.high,
        max_skip_frames=args.max_skip_frames,
        force_gpu_interval=args.force_gpu_interval,
        uncertainty_threshold=args.uncertainty_threshold,
        prediction_error_threshold=args.prediction_error_threshold,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    runtime_channels: List[str] = []
    runtime_f1: List[float] = []
    threshold_channels: List[str] = []
    threshold_f1: List[float] = []
    oracle_f1: List[float] = []
    target_channels: List[str] = []
    forced_reasons: Counter[str] = Counter()

    for frame_info in payload.get("frames", []):
        features = frame_info.get("selector_features")
        if not isinstance(features, dict):
            continue
        frame_id = int(frame_info.get("frame_id", 0))
        selector_input = ChannelSelectorInput(
            frame_features=frame_from_features(frame_id, features),
            state_snapshot=state_from_features(frame_id, features),
        )
        decision = selector.decide(selector_input)
        runtime_channel = ACTION_TO_CHANNEL.get(decision.action, "inference")
        threshold_channel = fixed_threshold_channel(features, args.low, args.high)
        target_channel = str(frame_info.get("best_channel", ""))

        runtime_channels.append(runtime_channel)
        runtime_f1.append(channel_f1(frame_info, runtime_channel))
        threshold_channels.append(threshold_channel)
        threshold_f1.append(channel_f1(frame_info, threshold_channel))
        oracle_f1.append(max(channel_f1(frame_info, name) for name in ("kalman", "gmc", "inference")))
        if target_channel:
            target_channels.append(target_channel)
        if decision.forced_reason:
            forced_reasons[decision.forced_reason] += 1

    return {
        "file": str(path),
        "model_available": selector.model_available,
        "load_error": selector.load_error,
        "runtime": summarize("runtime_selector", runtime_channels, runtime_f1),
        "fixed_threshold": summarize("fixed_frame_diff_threshold", threshold_channels, threshold_f1),
        "oracle": {
            "avg_f1": float(np.mean(oracle_f1)) if oracle_f1 else 0.0,
            "best_channel_counts": dict(Counter(target_channels)),
        },
        "forced_reasons": dict(forced_reasons),
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def weighted_avg(key: str, method: str) -> float:
        total_frames = sum(item[method]["frames"] for item in results)
        if total_frames <= 0:
            return 0.0
        return sum(item[method][key] * item[method]["frames"] for item in results) / total_frames

    return {
        "files": len(results),
        "runtime_avg_f1": weighted_avg("avg_f1", "runtime"),
        "runtime_avg_compute": weighted_avg("avg_compute", "runtime"),
        "runtime_gpu_ratio": weighted_avg("gpu_ratio", "runtime"),
        "fixed_avg_f1": weighted_avg("avg_f1", "fixed_threshold"),
        "fixed_avg_compute": weighted_avg("avg_compute", "fixed_threshold"),
        "fixed_gpu_ratio": weighted_avg("gpu_ratio", "fixed_threshold"),
    }


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.data)
    results = [evaluate_file(path, args) for path in paths]
    summary = aggregate(results)

    output = {
        "checkpoint": args.checkpoint,
        "summary": summary,
        "files": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

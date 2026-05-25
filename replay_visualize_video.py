#!/usr/bin/env python3
"""Render a replay visualization video from a selector label JSON."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from channel_selector_interfaces import (
    ChannelSelectorInput,
    FrameFeatureSnapshot,
    StateSnapshot,
)
from channel_selector_runtime import ACTION_TO_CHANNEL, RuntimeChannelSelector

COLORS_BGR = {
    "kalman":    (80, 200, 80),
    "gmc":       (220, 150, 40),
    "inference": (60, 100, 230),
}
LABEL_TEXT = {
    "kalman":    "KALMAN",
    "gmc":       "GMC",
    "inference": "YOLO",
}
BG = (20, 15, 28)
GRID = (45, 40, 55)
WHITE = (230, 230, 230)
GRAY = (130, 120, 145)
W, H = 1280, 720
FPS = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/labels/drone__M0201.json")
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--output", default="results/replay_vis_M0201.mp4")
    p.add_argument("--warmup-frames", type=int, default=10)
    p.add_argument("--max-skip-frames", type=int, default=10)
    p.add_argument("--force-gpu-interval", type=int, default=10)
    p.add_argument("--uncertainty-threshold", type=float, default=0.0)
    p.add_argument("--prediction-error-threshold", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--speed", type=int, default=2, help="Playback speed multiplier (write every Nth frame).")
    return p.parse_args()


def frame_to_input(frame_id: int, feat: Dict) -> ChannelSelectorInput:
    ff = FrameFeatureSnapshot(
        frame_id=frame_id,
        frame_diff_mean=float(feat.get("frame_diff_mean", 0) or 0),
        frame_diff_std=float(feat.get("frame_diff_std", 0) or 0),
        global_motion_dx=float(feat.get("global_motion_dx", 0) or 0),
        global_motion_dy=float(feat.get("global_motion_dy", 0) or 0),
        optical_flow_valid_ratio=float(feat.get("optical_flow_valid_ratio", 0) or 0),
        optical_flow_residual=float(feat.get("optical_flow_residual", 0) or 0),
        tracked_point_count=int(feat.get("tracked_point_count", 0) or 0),
        is_bootstrap_frame=bool(feat.get("is_bootstrap_frame", False)),
    )
    fsg = int(feat.get("frames_since_last_gpu", -1) or -1)
    ss = StateSnapshot(
        frame_id=frame_id,
        last_gpu_frame_id=frame_id - fsg if fsg >= 0 else -1,
        last_gpu_box_count=int(feat.get("last_gpu_box_count", 0) or 0),
        last_gpu_max_conf=float(feat.get("last_gpu_max_conf", 0) or 0),
        recent_gpu_box_count_mean=float(feat.get("recent_gpu_box_count_mean", 0) or 0),
        recent_gpu_box_count_delta=float(feat.get("recent_gpu_box_count_delta", 0) or 0),
        tracker_count=int(feat.get("tracker_count", 0) or 0),
        confirmed_tracker_count=int(feat.get("confirmed_tracker_count", 0) or 0),
        mean_track_age=float(feat.get("mean_track_age", 0) or 0),
        mean_time_since_update=float(feat.get("mean_time_since_update", 0) or 0),
        max_time_since_update=int(feat.get("max_time_since_update", 0) or 0),
        mean_speed=float(feat.get("mean_speed", 0) or 0),
        mean_position_uncertainty=float(feat.get("mean_position_uncertainty", 0) or 0),
        max_position_uncertainty=float(feat.get("max_position_uncertainty", 0) or 0),
        prediction_error_ma=float(feat.get("prediction_error_ma", 0) or 0),
        prediction_error_p95=float(feat.get("prediction_error_p95", 0) or 0),
    )
    return ChannelSelectorInput(frame_features=ff, state_snapshot=ss)


def draw_bar(img: np.ndarray, x: int, y: int, w: int, h: int,
             val: float, color: tuple, max_val: float = 1.0) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, -1)
    fill = int(w * min(val / max(max_val, 1e-6), 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, 1)


def draw_text(img, text, x, y, color=WHITE, scale=0.55, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_frame(
    canvas: np.ndarray,
    frame_id: int,
    total: int,
    channel: str,
    forced: bool,
    forced_reason: str,
    probs: Dict[str, float],
    feat: Dict,
    history: deque,
    counts: Dict[str, int],
    gpu_ratio_history: deque,
) -> None:
    canvas[:] = BG

    color = COLORS_BGR[channel]

    # ── Top banner ────────────────────────────────────────────────────────────
    banner_h = 90
    cv2.rectangle(canvas, (0, 0), (W, banner_h), tuple(max(0, c - 60) for c in color), -1)
    label = LABEL_TEXT[channel]
    tag = f"  [{forced_reason}]" if forced else "  [model]"
    draw_text(canvas, f"Frame {frame_id:05d} / {total}   →   {label}{tag}",
              20, 58, color=color, scale=1.6, thickness=2)

    # progress bar
    prog_y = banner_h - 8
    cv2.rectangle(canvas, (0, prog_y), (W, banner_h), (30, 25, 40), -1)
    prog_w = int(W * frame_id / max(total - 1, 1))
    cv2.rectangle(canvas, (0, prog_y), (prog_w, banner_h), color, -1)

    # ── Timeline strip (bottom half of banner area) ───────────────────────────
    strip_y = banner_h + 4
    strip_h = 28
    strip_w = W - 40
    cell_w = max(1, strip_w // max(len(history), 1))
    for i, ch in enumerate(history):
        cx = 20 + i * cell_w
        cv2.rectangle(canvas, (cx, strip_y), (cx + cell_w - 1, strip_y + strip_h),
                      COLORS_BGR[ch], -1)
    cv2.rectangle(canvas, (20, strip_y), (20 + strip_w, strip_y + strip_h), GRID, 1)
    draw_text(canvas, "recent decisions →", 20, strip_y - 4, GRAY, 0.38)

    # ── Left column: probabilities + counts ───────────────────────────────────
    lx, ly = 30, strip_y + strip_h + 30
    draw_text(canvas, "GRU probabilities", lx, ly, GRAY, 0.45)
    bar_w, bar_h = 260, 18
    for ch in ("kalman", "gmc", "inference"):
        ly += 28
        prob = probs.get(ch, 0.0)
        draw_bar(canvas, lx, ly - bar_h + 4, bar_w, bar_h, prob, COLORS_BGR[ch], 1.0)
        draw_text(canvas, f"{LABEL_TEXT[ch]:<9} {prob*100:5.1f}%", lx + bar_w + 10, ly, WHITE, 0.45)

    ly += 45
    draw_text(canvas, "channel counts (this replay)", lx, ly, GRAY, 0.45)
    total_so_far = max(sum(counts.values()), 1)
    for ch in ("kalman", "gmc", "inference"):
        ly += 28
        n = counts.get(ch, 0)
        pct = 100 * n / total_so_far
        draw_bar(canvas, lx, ly - bar_h + 4, bar_w, bar_h, pct, COLORS_BGR[ch], 100.0)
        draw_text(canvas, f"{LABEL_TEXT[ch]:<9} {pct:5.1f}%  ({n})", lx + bar_w + 10, ly, WHITE, 0.45)

    # ── Middle column: frame features ─────────────────────────────────────────
    mx = 430
    my = strip_y + strip_h + 30
    draw_text(canvas, "frame features", mx, my, GRAY, 0.45)
    feat_items = [
        ("frame_diff_mean",        float(feat.get("frame_diff_mean", 0) or 0),        1.0),
        ("frame_diff_std",         float(feat.get("frame_diff_std", 0) or 0),         0.5),
        ("optical_flow_valid",     float(feat.get("optical_flow_valid_ratio", 0) or 0), 1.0),
        ("optical_flow_residual",  float(feat.get("optical_flow_residual", 0) or 0),  5.0),
        ("global_motion_dx",       abs(float(feat.get("global_motion_dx", 0) or 0)),  20.0),
        ("global_motion_dy",       abs(float(feat.get("global_motion_dy", 0) or 0)),  20.0),
        ("tracked_points",         float(feat.get("tracked_point_count", 0) or 0),    200.0),
    ]
    bar_w2 = 220
    for name, val, mx_val in feat_items:
        my += 28
        draw_bar(canvas, mx, my - bar_h + 4, bar_w2, bar_h, val, (100, 160, 200), mx_val)
        draw_text(canvas, f"{name:<24} {val:7.3f}", mx + bar_w2 + 8, my, WHITE, 0.40)

    # tracker state
    my += 40
    draw_text(canvas, "tracker state", mx, my, GRAY, 0.45)
    tracker_items = [
        ("tracker_count",          float(feat.get("tracker_count", 0) or 0),           20.0),
        ("confirmed_trackers",     float(feat.get("confirmed_tracker_count", 0) or 0), 20.0),
        ("mean_speed",             float(feat.get("mean_speed", 0) or 0),              10.0),
        ("max_uncertainty",        float(feat.get("max_position_uncertainty", 0) or 0), 10000.0),
        ("pred_error_p95",         float(feat.get("prediction_error_p95", 0) or 0),    2.0),
        ("frames_since_gpu",       float(feat.get("frames_since_last_gpu", 0) or 0),   60.0),
    ]
    for name, val, mx_val in tracker_items:
        my += 28
        draw_bar(canvas, mx, my - bar_h + 4, bar_w2, bar_h, val, (180, 120, 80), mx_val)
        draw_text(canvas, f"{name:<24} {val:7.1f}", mx + bar_w2 + 8, my, WHITE, 0.40)

    # ── Right column: rolling GPU% sparkline ──────────────────────────────────
    rx = 950
    ry_top = strip_y + strip_h + 30
    spark_h = 180
    spark_w = W - rx - 30
    cv2.rectangle(canvas, (rx, ry_top), (rx + spark_w, ry_top + spark_h), GRID, 1)
    draw_text(canvas, "rolling GPU% (last 60 frames)", rx, ry_top - 6, GRAY, 0.40)

    vals = list(gpu_ratio_history)
    if len(vals) > 1:
        for i in range(len(vals) - 1):
            x1 = rx + int(i * spark_w / max(len(vals) - 1, 1))
            x2 = rx + int((i + 1) * spark_w / max(len(vals) - 1, 1))
            y1 = ry_top + spark_h - int(vals[i] * spark_h)
            y2 = ry_top + spark_h - int(vals[i + 1] * spark_h)
            cv2.line(canvas, (x1, y1), (x2, y2), COLORS_BGR["inference"], 2)

    # 50% guide line
    guide_y = ry_top + spark_h // 2
    cv2.line(canvas, (rx, guide_y), (rx + spark_w, guide_y), GRID, 1)
    draw_text(canvas, "50%", rx + spark_w + 3, guide_y + 5, GRAY, 0.35)

    current_ratio = vals[-1] if vals else 0.0
    draw_text(canvas, f"now: {current_ratio*100:.0f}%", rx, ry_top + spark_h + 20, WHITE, 0.50)

    # overall avg
    overall_gpu = counts.get("inference", 0) / max(total_so_far, 1)
    draw_text(canvas, f"avg: {overall_gpu*100:.1f}%", rx, ry_top + spark_h + 48, WHITE, 0.50)

    # ── Legend ────────────────────────────────────────────────────────────────
    leg_y = H - 30
    for i, (ch, lbl) in enumerate(LABEL_TEXT.items()):
        lx2 = rx + i * 100
        cv2.rectangle(canvas, (lx2, leg_y - 12), (lx2 + 14, leg_y + 2), COLORS_BGR[ch], -1)
        draw_text(canvas, lbl, lx2 + 18, leg_y, WHITE, 0.42)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(data_path.read_text())
    frames_raw = [f for f in payload.get("frames", []) if isinstance(f.get("selector_features"), dict)]
    total = len(frames_raw)
    print(f"Frames: {total}")

    selector = RuntimeChannelSelector(
        checkpoint_path=args.checkpoint,
        warmup_frames=args.warmup_frames,
        max_skip_frames=args.max_skip_frames,
        force_gpu_interval=args.force_gpu_interval,
        uncertainty_threshold=args.uncertainty_threshold,
        prediction_error_threshold=args.prediction_error_threshold,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (W, H))

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    history: deque = deque(maxlen=200)
    gpu_ratio_history: deque = deque(maxlen=60)
    counts: Dict[str, int] = {"kalman": 0, "gmc": 0, "inference": 0}
    recent_gpu: deque = deque(maxlen=60)

    for idx, fr in enumerate(frames_raw):
        feat = fr["selector_features"]
        fid = int(fr.get("frame_id", idx))
        sel_inp = frame_to_input(fid, feat)
        decision = selector.decide(sel_inp)
        channel = ACTION_TO_CHANNEL.get(decision.action, "inference")

        history.append(channel)
        counts[channel] = counts.get(channel, 0) + 1
        recent_gpu.append(1.0 if channel == "inference" else 0.0)
        gpu_ratio_history.append(float(np.mean(recent_gpu)))

        if idx % args.speed == 0:
            render_frame(
                canvas, fid, total, channel,
                bool(decision.forced_reason), decision.forced_reason or "",
                decision.probabilities, feat,
                history, counts, gpu_ratio_history,
            )
            writer.write(canvas)

        if idx % 100 == 0:
            print(f"  {idx}/{total}", end="\r")

    writer.release()
    print(f"\nSaved: {output_path}")
    k = counts.get("kalman", 0)
    g = counts.get("gmc", 0)
    inf = counts.get("inference", 0)
    n = max(k + g + inf, 1)
    print(f"Kalman {k/n*100:.1f}%  GMC {g/n*100:.1f}%  YOLO {inf/n*100:.1f}%")


if __name__ == "__main__":
    main()

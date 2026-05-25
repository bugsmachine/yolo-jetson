#!/usr/bin/env python3
"""Visualize per-frame channel selector decisions vs fixed threshold vs oracle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from channel_selector_interfaces import (
    ChannelSelectorInput,
    FrameFeatureSnapshot,
    StateSnapshot,
)
from channel_selector_runtime import ACTION_TO_CHANNEL, RuntimeChannelSelector

CHANNEL_COLORS = {"kalman": "#4CAF50", "gmc": "#2196F3", "inference": "#FF7043"}
CHANNEL_IDX = {"kalman": 0, "gmc": 1, "inference": 2}
COMPUTE_COST = {"kalman": 0.05, "gmc": 0.15, "inference": 1.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/labels/drone__M0201.json")
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--output", default="results/selector_vis.png")
    p.add_argument("--warmup-frames", type=int, default=10)
    p.add_argument("--max-skip-frames", type=int, default=60)
    p.add_argument("--force-gpu-interval", type=int, default=60)
    p.add_argument("--uncertainty-threshold", type=float, default=0.0)
    p.add_argument("--prediction-error-threshold", type=float, default=0.0)
    p.add_argument("--low", type=float, default=0.3)
    p.add_argument("--high", type=float, default=0.7)
    p.add_argument("--window", type=int, default=30, help="Rolling window for ratio/F1 plots.")
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


def fixed_channel(feat: Dict, low: float, high: float) -> str:
    if bool(feat.get("is_bootstrap_frame", False)):
        return "inference"
    score = float(feat.get("frame_diff_mean", 0) or 0)
    if score < low:
        return "kalman"
    if score < high:
        return "gmc"
    return "inference"


def channel_f1(frame_info: Dict, ch: str) -> float:
    ch_info = frame_info.get("channels", {}).get(ch, {})
    metrics = ch_info.get("metrics", {}) if isinstance(ch_info, dict) else {}
    return float(metrics.get("f1", 0) or 0)


def rolling(arr: np.ndarray, w: int) -> np.ndarray:
    result = np.empty(len(arr))
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        result[i] = arr[lo : i + 1].mean()
    return result


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(data_path.read_text())
    frames_raw = payload.get("frames", [])

    selector = RuntimeChannelSelector(
        checkpoint_path=args.checkpoint,
        warmup_frames=args.warmup_frames,
        max_skip_frames=args.max_skip_frames,
        force_gpu_interval=args.force_gpu_interval,
        uncertainty_threshold=args.uncertainty_threshold,
        prediction_error_threshold=args.prediction_error_threshold,
    )

    frame_ids: List[int] = []
    runtime_ch: List[str] = []
    fixed_ch: List[str] = []
    oracle_ch: List[str] = []
    runtime_f1: List[float] = []
    fixed_f1: List[float] = []
    oracle_f1: List[float] = []
    forced_reasons: Counter = Counter()
    is_forced: List[bool] = []

    for fr in frames_raw:
        feat = fr.get("selector_features")
        if not isinstance(feat, dict):
            continue
        fid = int(fr.get("frame_id", 0))
        sel_inp = frame_to_input(fid, feat)
        decision = selector.decide(sel_inp)
        rch = ACTION_TO_CHANNEL.get(decision.action, "inference")
        fch = fixed_channel(feat, args.low, args.high)
        best = str(fr.get("best_channel", ""))
        och = best if best in CHANNEL_IDX else max(
            ("kalman", "gmc", "inference"), key=lambda c: channel_f1(fr, c)
        )

        frame_ids.append(fid)
        runtime_ch.append(rch)
        fixed_ch.append(fch)
        oracle_ch.append(och)
        runtime_f1.append(channel_f1(fr, rch))
        fixed_f1.append(channel_f1(fr, fch))
        oracle_f1.append(channel_f1(fr, och))
        is_forced.append(bool(decision.forced_reason))
        if decision.forced_reason:
            forced_reasons[decision.forced_reason] += 1

    n = len(frame_ids)
    xs = np.arange(n)

    # Build numeric arrays for timeline heat map
    runtime_arr = np.array([CHANNEL_IDX[c] for c in runtime_ch])
    fixed_arr = np.array([CHANNEL_IDX[c] for c in fixed_ch])
    oracle_arr = np.array([CHANNEL_IDX[c] for c in oracle_ch])

    runtime_gpu = np.array([1 if c == "inference" else 0 for c in runtime_ch], dtype=float)
    fixed_gpu = np.array([1 if c == "inference" else 0 for c in fixed_ch], dtype=float)

    roll_runtime_gpu = rolling(runtime_gpu, args.window) * 100
    roll_fixed_gpu = rolling(fixed_gpu, args.window) * 100
    roll_runtime_f1 = rolling(np.array(runtime_f1), args.window)
    roll_fixed_f1 = rolling(np.array(fixed_f1), args.window)
    roll_oracle_f1 = rolling(np.array(oracle_f1), args.window)

    avg_runtime_compute = float(np.mean([COMPUTE_COST[c] for c in runtime_ch]))
    avg_fixed_compute = float(np.mean([COMPUTE_COST[c] for c in fixed_ch]))
    avg_runtime_f1 = float(np.mean(runtime_f1))
    avg_fixed_f1 = float(np.mean(fixed_f1))
    avg_oracle_f1 = float(np.mean(oracle_f1))

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#1a1a2e")
    title = data_path.stem
    fig.suptitle(
        f"Channel Selector — {title}",
        fontsize=14, fontweight="bold", color="white", y=0.98,
    )

    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.2, 1.2, 1.8],
                          hspace=0.55, wspace=0.3,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)

    cmap = matplotlib.colors.ListedColormap(
        [CHANNEL_COLORS["kalman"], CHANNEL_COLORS["gmc"], CHANNEL_COLORS["inference"]]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    def style_ax(ax, title):
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.set_title(title, color="#aaa", fontsize=9, loc="left", pad=3)
        ax.set_xlim(0, n)

    # Row 0: timeline strip plots (3 rows stacked = 3 imshow slices)
    ax_tl = fig.add_subplot(gs[0, :])
    timeline_data = np.vstack([runtime_arr, fixed_arr, oracle_arr])
    ax_tl.imshow(timeline_data, aspect="auto", cmap=cmap, norm=norm,
                 interpolation="nearest", extent=[0, n, -0.5, 2.5])
    # mark forced frames on runtime row
    forced_xs = [i for i, f in enumerate(is_forced) if f]
    if forced_xs:
        ax_tl.scatter(forced_xs, [2.0] * len(forced_xs),
                      marker="|", s=10, color="white", alpha=0.5, linewidths=0.4)
    ax_tl.set_yticks([0, 1, 2])
    ax_tl.set_yticklabels(["Oracle", "Fixed\nThresh", "Runtime"], fontsize=8, color="white")
    ax_tl.set_xlabel("Frame", color="white", fontsize=8)
    ax_tl.tick_params(colors="white", labelsize=8)
    for sp in ax_tl.spines.values():
        sp.set_color("#444")
    ax_tl.set_facecolor("#0d0d1a")
    ax_tl.set_title("Per-frame channel decision  (green=Kalman  blue=GMC  orange=GPU)  |=guardrail forced",
                     color="#aaa", fontsize=9, loc="left", pad=3)
    patches = [mpatches.Patch(color=CHANNEL_COLORS[c], label=c.capitalize()) for c in ("kalman", "gmc", "inference")]
    ax_tl.legend(handles=patches, loc="upper right", fontsize=8,
                 facecolor="#222", edgecolor="#555", labelcolor="white")

    # Row 1: rolling GPU ratio
    ax_gpu = fig.add_subplot(gs[1, :])
    style_ax(ax_gpu, f"Rolling GPU call rate (window={args.window} frames)")
    ax_gpu.plot(xs, roll_runtime_gpu, color="#FF7043", lw=1.2, label=f"Runtime ({runtime_gpu.mean()*100:.1f}% avg)")
    ax_gpu.plot(xs, roll_fixed_gpu, color="#90CAF9", lw=1.0, linestyle="--",
                label=f"Fixed thresh ({fixed_gpu.mean()*100:.1f}% avg)")
    ax_gpu.axhline(runtime_gpu.mean() * 100, color="#FF7043", lw=0.5, linestyle=":")
    ax_gpu.set_ylabel("GPU%", color="white", fontsize=8)
    ax_gpu.set_ylim(-5, 105)
    ax_gpu.legend(fontsize=8, facecolor="#222", edgecolor="#555", labelcolor="white")

    # Row 2: rolling F1
    ax_f1 = fig.add_subplot(gs[2, :])
    style_ax(ax_f1, f"Rolling F1 (window={args.window} frames)")
    ax_f1.plot(xs, roll_oracle_f1, color="#A5D6A7", lw=1.0, linestyle="--",
               label=f"Oracle ({avg_oracle_f1:.3f})")
    ax_f1.plot(xs, roll_runtime_f1, color="#FF7043", lw=1.2,
               label=f"Runtime ({avg_runtime_f1:.3f})")
    ax_f1.plot(xs, roll_fixed_f1, color="#90CAF9", lw=1.0, linestyle="--",
               label=f"Fixed ({avg_fixed_f1:.3f})")
    ax_f1.set_ylabel("F1", color="white", fontsize=8)
    ax_f1.set_ylim(-0.05, 1.05)
    ax_f1.legend(fontsize=8, facecolor="#222", edgecolor="#555", labelcolor="white")

    # Row 3 left: channel distribution bars
    ax_bar = fig.add_subplot(gs[3, 0])
    style_ax(ax_bar, "Channel distribution")
    methods = ["Runtime", "Fixed", "Oracle"]
    kalman_pcts = [
        runtime_ch.count("kalman") / n * 100,
        fixed_ch.count("kalman") / n * 100,
        oracle_ch.count("kalman") / n * 100,
    ]
    gmc_pcts = [
        runtime_ch.count("gmc") / n * 100,
        fixed_ch.count("gmc") / n * 100,
        oracle_ch.count("gmc") / n * 100,
    ]
    inf_pcts = [
        runtime_ch.count("inference") / n * 100,
        fixed_ch.count("inference") / n * 100,
        oracle_ch.count("inference") / n * 100,
    ]
    bx = np.arange(3)
    w = 0.55
    ax_bar.bar(bx, kalman_pcts, w, color=CHANNEL_COLORS["kalman"], label="Kalman")
    ax_bar.bar(bx, gmc_pcts, w, bottom=kalman_pcts, color=CHANNEL_COLORS["gmc"], label="GMC")
    ax_bar.bar(bx, inf_pcts, w,
               bottom=[k + g for k, g in zip(kalman_pcts, gmc_pcts)],
               color=CHANNEL_COLORS["inference"], label="GPU")
    ax_bar.set_xticks(bx)
    ax_bar.set_xticklabels(methods, color="white", fontsize=9)
    ax_bar.set_ylabel("%", color="white", fontsize=8)
    ax_bar.set_ylim(0, 110)
    ax_bar.legend(fontsize=8, facecolor="#222", edgecolor="#555", labelcolor="white")
    for i, (k, g, inf) in enumerate(zip(kalman_pcts, gmc_pcts, inf_pcts)):
        if k > 4:
            ax_bar.text(i, k / 2, f"{k:.0f}%", ha="center", va="center", fontsize=7, color="white")
        if g > 4:
            ax_bar.text(i, k + g / 2, f"{g:.0f}%", ha="center", va="center", fontsize=7, color="white")
        if inf > 4:
            ax_bar.text(i, k + g + inf / 2, f"{inf:.0f}%", ha="center", va="center", fontsize=7, color="white")

    # Row 3 right: summary stats + forced reasons
    ax_stats = fig.add_subplot(gs[3, 1])
    ax_stats.set_facecolor("#0d0d1a")
    ax_stats.axis("off")
    ax_stats.set_title("Summary", color="#aaa", fontsize=9, loc="left", pad=3)

    lines = [
        ("", ""),
        ("", "Runtime vs Fixed Threshold"),
        ("Avg F1", f"{avg_runtime_f1:.3f}  vs  {avg_fixed_f1:.3f}  (oracle {avg_oracle_f1:.3f})"),
        ("Avg compute", f"{avg_runtime_compute:.3f}  vs  {avg_fixed_compute:.3f}"),
        ("GPU ratio", f"{runtime_gpu.mean()*100:.1f}%  vs  {fixed_gpu.mean()*100:.1f}%"),
        ("", ""),
        ("", "Guardrail forced reasons"),
    ]
    for reason, cnt in sorted(forced_reasons.items(), key=lambda x: -x[1]):
        lines.append((reason, str(cnt)))
    lines.append(("total forced", str(sum(forced_reasons.values()))))
    lines.append(("total frames", str(n)))

    y = 0.97
    for label, val in lines:
        if label == "":
            ax_stats.text(0.02, y, val, transform=ax_stats.transAxes,
                          fontsize=8.5, color="#aaa", va="top", fontstyle="italic")
        else:
            ax_stats.text(0.02, y, label + ":", transform=ax_stats.transAxes,
                          fontsize=8, color="#888", va="top")
            ax_stats.text(0.45, y, val, transform=ax_stats.transAxes,
                          fontsize=8, color="white", va="top")
        y -= 0.10

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

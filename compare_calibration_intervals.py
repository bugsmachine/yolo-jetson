#!/usr/bin/env python3
"""Compare forced YOLO calibration intervals against per-frame YOLO reference.

The reference is YOLO on every frame. For each interval, the runtime selector
may choose Kalman/GMC/YOLO, with guard rails forcing YOLO at least every N
frames. Kalman and GMC predictions are evaluated against the same frame's
reference detections.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from channel_selector_interfaces import (
    ChannelSelectorInput,
    ReaderFeatureExtractor,
    StateSnapshot,
)
from kalman_tracker import KalmanTrackerConfig, MultiObjectKalmanTracker
from src.runtime_selector import ACTION_TO_CHANNEL, RuntimeChannelSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forced calibration intervals.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--yolo-model", default="/home/tan/Desktop/all/yolo11m_int8.engine")
    parser.add_argument("--intervals", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--output", default="results/calibration_interval_eval.json")
    parser.add_argument("--render-dir", default="results/calibration_interval_videos")
    parser.add_argument("--render", action="store_true", help="Render one boxed MP4 per interval.")
    parser.add_argument("--simulate-yolo", action="store_true", help="Use deterministic fake YOLO boxes for non-Jetson tests.")
    return parser.parse_args()


def load_frames(video_path: str, max_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: List[np.ndarray] = []
    while max_frames <= 0 or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (640, 640)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return frames


def parse_yolo_result(result: Any) -> List[Dict[str, float]]:
    boxes: List[Dict[str, float]] = []
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        boxes.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(box.conf[0].cpu().numpy()),
                "class": int(box.cls[0].cpu().numpy()),
            }
        )
    return boxes


def simulated_yolo_boxes(frame_id: int, frame: np.ndarray) -> List[Dict[str, float]]:
    h, w = frame.shape[:2]
    x1 = int((w * 0.18 + frame_id * 3) % max(1, w - 180))
    y1 = int(h * 0.28)
    boxes = [
        {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(min(w - 1, x1 + int(w * 0.16))),
            "y2": float(min(h - 1, y1 + int(h * 0.24))),
            "conf": 0.72,
            "class": 0,
        }
    ]
    if frame_id % 3 == 0:
        x1b = int(w * 0.58)
        y1b = int((h * 0.18 + frame_id * 2) % max(1, h - 160))
        boxes.append(
            {
                "x1": float(x1b),
                "y1": float(y1b),
                "x2": float(min(w - 1, x1b + int(w * 0.13))),
                "y2": float(min(h - 1, y1b + int(h * 0.20))),
                "conf": 0.64,
                "class": 0,
            }
        )
    return boxes


def run_pure_yolo(
    frames: List[np.ndarray],
    model_path: str,
    imgsz: int,
    conf: float,
    iou: float,
    simulate: bool,
) -> Tuple[List[List[Dict[str, float]]], float]:
    if simulate:
        start = time.perf_counter()
        detections = [simulated_yolo_boxes(frame_id, frame) for frame_id, frame in enumerate(frames)]
        return detections, time.perf_counter() - start

    from ultralytics import YOLO

    model = YOLO(model_path)
    model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    detections: List[List[Dict[str, float]]] = []
    start = time.perf_counter()
    for frame in frames:
        result = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        detections.append(parse_yolo_result(result))
    return detections, time.perf_counter() - start


def box_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1["x2"] - box1["x1"]) * max(0.0, box1["y2"] - box1["y1"])
    area2 = max(0.0, box2["x2"] - box2["x1"]) * max(0.0, box2["y2"] - box2["y1"])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def compare_boxes(
    pred_boxes: List[Dict[str, float]],
    ref_boxes: List[Dict[str, float]],
    match_iou: float,
) -> Dict[str, float]:
    pairs: List[Tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(pred_boxes):
        for ref_idx, ref in enumerate(ref_boxes):
            iou = box_iou(pred, ref)
            if iou >= match_iou:
                pairs.append((iou, pred_idx, ref_idx))
    pairs.sort(reverse=True)

    used_pred = set()
    used_ref = set()
    matched_ious: List[float] = []
    for iou, pred_idx, ref_idx in pairs:
        if pred_idx in used_pred or ref_idx in used_ref:
            continue
        used_pred.add(pred_idx)
        used_ref.add(ref_idx)
        matched_ious.append(iou)

    tp = len(matched_ious)
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(ref_boxes) - tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "pred_count": float(len(pred_boxes)),
        "ref_count": float(len(ref_boxes)),
    }


def state_snapshot_from_tracker(
    frame_id: int,
    last_yolo_frame_id: int,
    last_yolo_boxes: List[Dict[str, float]],
    recent_box_counts: deque,
    tracker: MultiObjectKalmanTracker,
) -> StateSnapshot:
    summary = tracker.get_state_summary()
    recent_mean = float(np.mean(recent_box_counts)) if recent_box_counts else 0.0
    recent_delta = float(recent_box_counts[-1] - recent_box_counts[-2]) if len(recent_box_counts) >= 2 else 0.0
    max_conf = max((float(box.get("conf", 0.0)) for box in last_yolo_boxes), default=0.0)
    return StateSnapshot(
        frame_id=frame_id,
        last_gpu_frame_id=last_yolo_frame_id,
        last_gpu_source="pure_yolo" if last_yolo_frame_id >= 0 else "",
        last_gpu_box_count=len(last_yolo_boxes),
        last_gpu_max_conf=max_conf,
        recent_gpu_box_count_mean=recent_mean,
        recent_gpu_box_count_delta=recent_delta,
        tracker_count=int(summary["tracker_count"]),
        confirmed_tracker_count=int(summary["confirmed_tracker_count"]),
        mean_track_age=float(summary["mean_track_age"]),
        mean_time_since_update=float(summary["mean_time_since_update"]),
        max_time_since_update=int(summary["max_time_since_update"]),
        mean_speed=float(summary["mean_speed"]),
        mean_position_uncertainty=float(summary["mean_position_uncertainty"]),
        max_position_uncertainty=float(summary["max_position_uncertainty"]),
    )


def gmc_predict(
    latest_boxes: List[Dict[str, float]],
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> List[Dict[str, float]]:
    predicted: List[Dict[str, float]] = []
    for box in latest_boxes:
        x1 = max(0.0, min(box["x1"] + dx, width - 1.0))
        y1 = max(0.0, min(box["y1"] + dy, height - 1.0))
        x2 = max(0.0, min(box["x2"] + dx, width - 1.0))
        y2 = max(0.0, min(box["y2"] + dy, height - 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        predicted.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(box.get("conf", 0.0)) * 0.9,
                "class": int(box.get("class", 0)),
            }
        )
    return predicted


def visual_channel_style(channel: str) -> Tuple[str, Tuple[int, int, int]]:
    if channel == "kalman":
        return "KALMAN", (255, 0, 0)
    if channel == "gmc":
        return "GMC", (0, 165, 255)
    return "YOLO", (0, 220, 0)


def draw_visual_frame(
    frame: np.ndarray,
    frame_id: int,
    interval: int,
    channel: str,
    boxes: List[Dict[str, float]],
    frame_metric: Dict[str, float],
) -> np.ndarray:
    output = frame.copy()
    label_name, color = visual_channel_style(channel)
    for box in boxes:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        x1 = max(0, min(x1, output.shape[1] - 1))
        x2 = max(0, min(x2, output.shape[1] - 1))
        y1 = max(0, min(y1, output.shape[0] - 1))
        y2 = max(0, min(y2, output.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        text = f"{label_name} cls:{int(box.get('class', 0))} {float(box.get('conf', 0.0)):.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, max(0, y1 - th - 6)), (min(output.shape[1] - 1, x1 + tw), y1), color, -1)
        cv2.putText(output, text, (x1, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    header = (
        f"Frame:{frame_id} Interval:{interval} Channel:{label_name} "
        f"F1_vs_YOLO:{frame_metric['f1']:.2f} IoU:{frame_metric['mean_matched_iou']:.2f}"
    )
    cv2.putText(output, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return output


def evaluate_interval(
    interval: int,
    frames: List[np.ndarray],
    pure_yolo: List[List[Dict[str, float]]],
    checkpoint_path: str,
    match_iou: float,
    render_path: str = "",
) -> Dict[str, Any]:
    selector = RuntimeChannelSelector(
        checkpoint_path=checkpoint_path,
        max_skip_frames=interval,
        force_gpu_interval=interval,
        uncertainty_threshold=100000.0,
        prediction_error_threshold=0.9,
    )
    feature_extractor = ReaderFeatureExtractor()
    tracker = MultiObjectKalmanTracker(KalmanTrackerConfig(max_age=30, min_hits=2, iou_threshold=0.3))

    latest_yolo_boxes: List[Dict[str, float]] = []
    last_yolo_frame_id = -1
    recent_box_counts: deque = deque(maxlen=30)
    channel_counts: Counter[str] = Counter()
    forced_reasons: Counter[str] = Counter()
    metrics: List[Dict[str, float]] = []
    per_frame: List[Dict[str, Any]] = []
    writer = None
    start = time.perf_counter()

    for frame_id, frame in enumerate(frames):
        frame_features = feature_extractor.extract(frame, frame_id)
        state_snapshot = state_snapshot_from_tracker(
            frame_id, last_yolo_frame_id, latest_yolo_boxes, recent_box_counts, tracker
        )
        decision = selector.decide(ChannelSelectorInput(frame_features, state_snapshot))
        channel = ACTION_TO_CHANNEL.get(decision.action, "inference")
        if decision.forced_reason:
            forced_reasons[decision.forced_reason] += 1

        if channel == "inference":
            selected_boxes = [box.copy() for box in pure_yolo[frame_id]]
            latest_yolo_boxes = [box.copy() for box in selected_boxes]
            last_yolo_frame_id = frame_id
            recent_box_counts.append(len(selected_boxes))
            tracker.update(selected_boxes)
            visual_channel = "yolo"
        elif channel == "kalman":
            selected_boxes = tracker.predict()
            visual_channel = "kalman"
        else:
            h, w = frame.shape[:2]
            selected_boxes = gmc_predict(latest_yolo_boxes, frame_features.global_motion_dx, frame_features.global_motion_dy, w, h)
            visual_channel = "gmc"

        channel_counts[visual_channel] += 1
        frame_metric = compare_boxes(selected_boxes, pure_yolo[frame_id], match_iou)
        metrics.append(frame_metric)
        per_frame.append(
            {
                "frame_id": frame_id,
                "channel": visual_channel,
                "forced_reason": decision.forced_reason,
                "f1": frame_metric["f1"],
                "precision": frame_metric["precision"],
                "recall": frame_metric["recall"],
                "mean_matched_iou": frame_metric["mean_matched_iou"],
                "pred_count": frame_metric["pred_count"],
                "ref_count": frame_metric["ref_count"],
            }
        )
        if render_path:
            if writer is None:
                Path(render_path).parent.mkdir(parents=True, exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(render_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create render video: {render_path}")
            writer.write(draw_visual_frame(frame, frame_id, interval, visual_channel, selected_boxes, frame_metric))

    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - start

    def avg(key: str) -> float:
        return float(np.mean([item[key] for item in metrics])) if metrics else 0.0

    total = max(1, len(frames))
    return {
        "interval": interval,
        "frames": len(frames),
        "model_available": selector.model_available,
        "load_error": selector.load_error,
        "channel_counts": dict(channel_counts),
        "channel_ratios": {key: value / total for key, value in channel_counts.items()},
        "forced_reasons": dict(forced_reasons),
        "macro_frame_precision": avg("precision"),
        "macro_frame_recall": avg("recall"),
        "macro_frame_f1": avg("f1"),
        "mean_matched_iou": avg("mean_matched_iou"),
        "elapsed_s": elapsed,
        "fps": len(frames) / elapsed if elapsed > 0 else 0.0,
        "avg_pred_count": avg("pred_count"),
        "avg_ref_yolo_count": avg("ref_count"),
        "per_frame": per_frame,
    }


def main() -> None:
    args = parse_args()
    frames = load_frames(args.video, args.max_frames)
    print(f"Loaded {len(frames)} frames from {args.video}")
    pure_yolo, yolo_elapsed = run_pure_yolo(
        frames, args.yolo_model, args.imgsz, args.conf, args.iou, args.simulate_yolo
    )
    print(f"Pure YOLO reference done: {len(pure_yolo)} frames, {yolo_elapsed:.2f}s")

    results = []
    render_dir = Path(args.render_dir)
    for interval in args.intervals:
        render_path = str(render_dir / f"interval_{interval}.mp4") if args.render else ""
        results.append(
            evaluate_interval(
                interval=interval,
                frames=frames,
                pure_yolo=pure_yolo,
                checkpoint_path=args.checkpoint,
                match_iou=args.match_iou,
                render_path=render_path,
            )
        )

    output = {
        "video": args.video,
        "reference": {
            "model": args.yolo_model,
            "simulate_yolo": args.simulate_yolo,
            "frames": len(frames),
            "elapsed_s": yolo_elapsed,
            "avg_boxes": float(np.mean([len(boxes) for boxes in pure_yolo])) if pure_yolo else 0.0,
        },
        "match_iou": args.match_iou,
        "render_dir": str(render_dir) if args.render else "",
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\ninterval | F1    | precision | recall | mean IoU | eval FPS | YOLO% | GMC%  | Kalman%")
    print("-" * 89)
    for item in results:
        ratios = item["channel_ratios"]
        print(
            f"{item['interval']:>8} | "
            f"{item['macro_frame_f1']:.4f} | "
            f"{item['macro_frame_precision']:.4f}    | "
            f"{item['macro_frame_recall']:.4f} | "
            f"{item['mean_matched_iou']:.4f}   | "
            f"{item['fps']:8.1f} | "
            f"{ratios.get('yolo', 0.0) * 100:5.1f} | "
            f"{ratios.get('gmc', 0.0) * 100:5.1f} | "
            f"{ratios.get('kalman', 0.0) * 100:7.1f}"
        )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import INDEX_TO_CHANNEL, VideoSelectorDataset
from src.evaluate import build_model, compute_scheduler_metrics, format_confusion_matrix


CHANNEL_COLORS: dict[str, tuple[int, int, int]] = {
    "kalman": (255, 180, 40),
    "gmc": (40, 220, 255),
    "inference": (80, 255, 80),
}
TARGET_COLOR = (80, 80, 255)
DISPLAY_CHANNELS: tuple[str, ...] = ("kalman", "gmc", "yolo")
CHANNEL_DISPLAY_NAME: dict[str, str] = {
    "kalman": "kalman",
    "gmc": "gmc",
    "inference": "yolo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a selector checkpoint and render predicted-channel boxes on videos."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Path to checkpoint.")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Dataset split.")
    parser.add_argument("--output-dir", default="results/visualization", help="Directory for outputs.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config train.batch_size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for quick tests.")
    parser.add_argument("--videos", nargs="*", default=None, help="Optional video names to visualize.")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to render.")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS.")
    parser.add_argument("--draw-target", action="store_true", help="Also draw target-channel boxes.")
    parser.add_argument("--no-video", action="store_true", help="Only evaluate metrics; skip video rendering.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_video_label_paths(
    config: dict[str, Any],
    split: str,
    selected_videos: set[str] | None = None,
) -> tuple[list[Path], list[Path], list[str]]:
    data_cfg = config["data"]
    video_names = list(data_cfg[f"{split}_videos"])
    if selected_videos is not None:
        video_names = [name for name in video_names if name in selected_videos]
        missing_requested = sorted(selected_videos.difference(video_names))
        if missing_requested:
            raise ValueError(f"Requested videos are not in {split}_videos: {missing_requested}")

    video_dir = Path(data_cfg.get("video_dir", ""))
    label_dir = Path(data_cfg["label_dir"])
    video_paths = [video_dir / f"{name}.mp4" for name in video_names]
    label_paths = [label_dir / f"{name}.json" for name in video_names]
    missing = [str(path) for path in [*video_paths, *label_paths] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    return video_paths, label_paths, video_names


def create_dataset(
    config: dict[str, Any],
    split: str,
    checkpoint: dict[str, Any],
    selected_videos: set[str] | None = None,
) -> VideoSelectorDataset:
    data_cfg = config["data"]
    video_paths, label_paths, _ = resolve_video_label_paths(config, split, selected_videos)
    dataset = VideoSelectorDataset(
        video_paths=video_paths,
        label_paths=label_paths,
        cache_root=data_cfg["cache_dir"],
        n_frames=int(data_cfg["n_frames"]),
        margin=float(data_cfg["relabel_margin"]),
        cache_frames=False,
    )
    if checkpoint.get("feature_mean") is not None:
        dataset.set_feature_normalization(checkpoint["feature_mean"], checkpoint["feature_std"])
    return dataset


def predict_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset: VideoSelectorDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    records: list[dict[str, Any]] = []
    sample_index = 0

    with torch.no_grad():
        for inputs, batch_targets in tqdm(dataloader, desc="evaluate", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu()
            batch_predictions = probs.argmax(dim=1)

            for row_idx in range(inputs.shape[0]):
                sample = dataset.get_sample(sample_index)
                target_idx = int(batch_targets[row_idx].item())
                pred_idx = int(batch_predictions[row_idx].item())
                pred_channel = INDEX_TO_CHANNEL[pred_idx]
                target_channel = INDEX_TO_CHANNEL[target_idx]
                prob_map = {
                    INDEX_TO_CHANNEL[class_idx]: float(probs[row_idx, class_idx].item())
                    for class_idx in range(probs.shape[1])
                }
                targets.append(target_idx)
                predictions.append(pred_idx)
                records.append(
                    {
                        "sample_index": sample_index,
                        "video_name": sample.video_name,
                        "frame_id": sample.frame_id,
                        "predicted_label": pred_idx,
                        "predicted_channel": pred_channel,
                        "target_label": target_idx,
                        "target_channel": target_channel,
                        "correct": pred_idx == target_idx,
                        "confidence": prob_map[pred_channel],
                        "probabilities": prob_map,
                        "channel_f1": sample.channel_f1,
                    }
                )
                sample_index += 1

    return np.asarray(targets), np.asarray(predictions), records


def compute_classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    conf_mat = confusion_matrix(targets, predictions, labels=[0, 1, 2])
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)) if len(targets) else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "per_class": {
            INDEX_TO_CHANNEL[idx]: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx in range(3)
        },
    }
    return metrics, conf_mat


def display_channel_name(channel: str) -> str:
    return CHANNEL_DISPLAY_NAME.get(channel, channel)


def empty_channel_counts() -> dict[str, int]:
    return {channel: 0 for channel in DISPLAY_CHANNELS}


def channel_ratios(counts: dict[str, int], total: int) -> dict[str, float]:
    return {
        channel: (counts[channel] / total if total else 0.0)
        for channel in DISPLAY_CHANNELS
    }


def compute_per_video_channel_stats(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for record in predictions:
        video_name = str(record["video_name"])
        if video_name not in stats:
            stats[video_name] = {
                "num_samples": 0,
                "predicted_counts": empty_channel_counts(),
                "target_counts": empty_channel_counts(),
            }

        item = stats[video_name]
        item["num_samples"] += 1
        pred_channel = display_channel_name(str(record["predicted_channel"]))
        target_channel = display_channel_name(str(record["target_channel"]))
        item["predicted_counts"][pred_channel] += 1
        item["target_counts"][target_channel] += 1

    for item in stats.values():
        total = int(item["num_samples"])
        item["predicted_ratios"] = channel_ratios(item["predicted_counts"], total)
        item["target_ratios"] = channel_ratios(item["target_counts"], total)

    return dict(sorted(stats.items()))


def save_per_video_stats_csv(output_dir: Path, checkpoint_path: Path, split: str, stats: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"per_video_channel_ratios_{checkpoint_path.stem}_{split}.csv"
    fieldnames = [
        "video_name",
        "num_samples",
        "pred_kalman_count",
        "pred_gmc_count",
        "pred_yolo_count",
        "pred_kalman_ratio",
        "pred_gmc_ratio",
        "pred_yolo_ratio",
        "target_kalman_count",
        "target_gmc_count",
        "target_yolo_count",
        "target_kalman_ratio",
        "target_gmc_ratio",
        "target_yolo_ratio",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for video_name, item in stats.items():
            row = {
                "video_name": video_name,
                "num_samples": item["num_samples"],
            }
            for channel in DISPLAY_CHANNELS:
                row[f"pred_{channel}_count"] = item["predicted_counts"][channel]
                row[f"pred_{channel}_ratio"] = f"{item['predicted_ratios'][channel]:.6f}"
                row[f"target_{channel}_count"] = item["target_counts"][channel]
                row[f"target_{channel}_ratio"] = f"{item['target_ratios'][channel]:.6f}"
            writer.writerow(row)
    return output_path


def log_per_video_channel_stats(logger: logging.Logger, stats: dict[str, Any]) -> None:
    logger.info("Per-video predicted channel ratios:")
    for video_name, item in stats.items():
        ratios = item["predicted_ratios"]
        counts = item["predicted_counts"]
        logger.info(
            "%s | n=%d | kalman=%d(%.2f%%) gmc=%d(%.2f%%) yolo=%d(%.2f%%)",
            video_name,
            item["num_samples"],
            counts["kalman"],
            ratios["kalman"] * 100.0,
            counts["gmc"],
            ratios["gmc"] * 100.0,
            counts["yolo"],
            ratios["yolo"] * 100.0,
        )


def load_frames_by_id(label_path: Path) -> dict[int, dict[str, Any]]:
    with label_path.open("r", encoding="utf-8") as file:
        label_data = json.load(file)
    return {int(frame["frame_id"]): frame for frame in label_data.get("frames", [])}


def box_xyxy(box: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(round(float(box.get("x1", 0)))),
        int(round(float(box.get("y1", 0)))),
        int(round(float(box.get("x2", 0)))),
        int(round(float(box.get("y2", 0)))),
    )


def draw_boxes(
    frame: np.ndarray,
    boxes: list[dict[str, Any]],
    color: tuple[int, int, int],
    prefix: str,
    thickness: int = 2,
) -> None:
    height, width = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box_xyxy(box)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        conf = box.get("conf")
        label = prefix if conf is None else f"{prefix} {float(conf):.2f}"
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_header(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    frame_info: dict[str, Any] | None,
    pred_record: dict[str, Any] | None,
    draw_target: bool,
) -> None:
    if pred_record is None or frame_info is None:
        draw_header(frame, f"frame={frame_idx} no selector sample")
        return

    pred_channel = str(pred_record["predicted_channel"])
    target_channel = str(pred_record["target_channel"])
    pred_result = frame_info.get("channels", {}).get(pred_channel, {}).get("result", {})
    pred_boxes = pred_result.get("boxes", []) if isinstance(pred_result, dict) else []
    draw_boxes(frame, pred_boxes, CHANNEL_COLORS[pred_channel], pred_channel)

    if draw_target and target_channel != pred_channel:
        target_result = frame_info.get("channels", {}).get(target_channel, {}).get("result", {})
        target_boxes = target_result.get("boxes", []) if isinstance(target_result, dict) else []
        draw_boxes(frame, target_boxes, TARGET_COLOR, f"target:{target_channel}", thickness=1)

    draw_header(
        frame,
        f"frame={frame_idx} pred={pred_channel} target={target_channel} conf={pred_record['confidence']:.3f}",
    )


def render_cached_frames(
    cache_dir: Path,
    label_path: Path,
    output_path: Path,
    prediction_records: list[dict[str, Any]],
    fps: float,
    draw_target: bool,
) -> None:
    frame_paths = sorted(cache_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No cached frames found in {cache_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read cached frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")

    frame_labels = load_frames_by_id(label_path)
    predictions_by_frame = {int(item["frame_id"]): item for item in prediction_records}
    for frame_path in tqdm(frame_paths, desc=f"render {cache_dir.name}", leave=False):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        frame_idx = int(frame_path.stem)
        annotate_frame(
            frame=frame,
            frame_idx=frame_idx,
            frame_info=frame_labels.get(frame_idx),
            pred_record=predictions_by_frame.get(frame_idx),
            draw_target=draw_target,
        )
        writer.write(frame)

    writer.release()


def render_video(
    video_path: Path,
    label_path: Path,
    output_path: Path,
    prediction_records: list[dict[str, Any]],
    cache_dir: Path | None = None,
    fps_override: float | None = None,
    draw_target: bool = False,
) -> None:
    frame_labels = load_frames_by_id(label_path)
    predictions_by_frame = {int(item["frame_id"]): item for item in prediction_records}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if cache_dir is not None:
            render_cached_frames(
                cache_dir=cache_dir,
                label_path=label_path,
                output_path=output_path,
                prediction_records=prediction_records,
                fps=fps_override or 25.0,
                draw_target=draw_target,
            )
            return
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps_override or float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video writer: {output_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotate_frame(
            frame=frame,
            frame_idx=frame_idx,
            frame_info=frame_labels.get(frame_idx),
            pred_record=predictions_by_frame.get(frame_idx),
            draw_target=draw_target,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def save_payload(
    output_dir: Path,
    checkpoint_path: Path,
    split: str,
    classification_metrics: dict[str, Any],
    scheduler_metrics: dict[str, Any],
    per_video_channel_stats: dict[str, Any],
    confusion: np.ndarray,
    predictions: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_visual_{checkpoint_path.stem}_{split}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "split": split,
                "classification": classification_metrics,
                "scheduler": scheduler_metrics,
                "per_video_channel_stats": per_video_channel_stats,
                "confusion_matrix": confusion.tolist(),
                "predictions": predictions,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    return output_path


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("evaluate_visualize")

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("config"):
        config = checkpoint["config"]

    selected_videos = set(args.videos) if args.videos else None
    dataset = create_dataset(config, args.split, checkpoint, selected_videos)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: max(0, int(args.max_samples))]

    batch_size = args.batch_size or int(config["train"]["batch_size"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Evaluating %s on %s split with %d samples.", checkpoint_path, args.split, len(dataset))
    targets, predictions, prediction_records = predict_dataset(model, dataloader, dataset, device)
    classification_metrics, conf_mat = compute_classification_metrics(targets, predictions)
    scheduler_metrics = compute_scheduler_metrics(dataset, predictions, dict(config["eval"]["compute_cost"]))
    per_video_channel_stats = compute_per_video_channel_stats(prediction_records)

    logger.info(
        "Accuracy=%.4f Macro-F1=%.4f",
        classification_metrics["accuracy"],
        classification_metrics["macro_f1"],
    )
    logger.info("Confusion matrix:\n%s", format_confusion_matrix(conf_mat))
    logger.info(
        "Selector scheduler: avg_f1=%.4f compute=%.3f f1/compute=%.3f",
        scheduler_metrics["selector_avg_f1"],
        scheduler_metrics["selector_avg_compute"],
        scheduler_metrics["selector_f1_per_compute"],
    )
    log_per_video_channel_stats(logger, per_video_channel_stats)

    output_dir = Path(args.output_dir)
    channel_stats_path = save_per_video_stats_csv(output_dir, checkpoint_path, args.split, per_video_channel_stats)
    results_path = save_payload(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        split=args.split,
        classification_metrics=classification_metrics,
        scheduler_metrics=scheduler_metrics,
        per_video_channel_stats=per_video_channel_stats,
        confusion=conf_mat,
        predictions=prediction_records,
    )
    logger.info("Saved metrics and predictions to %s", results_path)
    logger.info("Saved per-video channel ratios to %s", channel_stats_path)

    if args.no_video:
        return

    records_by_video: dict[str, list[dict[str, Any]]] = {}
    for record in prediction_records:
        records_by_video.setdefault(str(record["video_name"]), []).append(record)

    video_paths, label_paths, video_names = resolve_video_label_paths(config, args.split, selected_videos)
    render_count = 0
    for video_name, video_path, label_path in zip(video_names, video_paths, label_paths):
        if video_name not in records_by_video:
            continue
        if args.max_videos is not None and render_count >= args.max_videos:
            break
        output_video = output_dir / f"{video_name}_{checkpoint_path.stem}_{args.split}_boxes.mp4"
        logger.info("Rendering %s", output_video)
        render_video(
            video_path=video_path,
            label_path=label_path,
            output_path=output_video,
            prediction_records=records_by_video[video_name],
            cache_dir=Path(config["data"]["cache_dir"]) / video_name,
            fps_override=args.fps,
            draw_target=args.draw_target,
        )
        render_count += 1
    logger.info("Rendered %d visualization video(s) to %s", render_count, output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import INDEX_TO_CHANNEL, VideoSelectorDataset
from src.model import VideoSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate video selector model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_paths(config: dict[str, Any], video_names: list[str]) -> tuple[list[Path], list[Path]]:
    video_dir = Path(config["data"].get("video_dir", ""))
    label_dir = Path(config["data"]["label_dir"])
    video_paths = [video_dir / f"{name}.mp4" for name in video_names]
    label_paths = [label_dir / f"{name}.json" for name in video_names]
    missing = [str(path) for path in label_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    return video_paths, label_paths


def create_val_dataset(config: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> VideoSelectorDataset:
    data_cfg = config["data"]
    video_paths, label_paths = resolve_paths(config, list(data_cfg["val_videos"]))
    dataset = VideoSelectorDataset(
        video_paths=video_paths,
        label_paths=label_paths,
        cache_root=data_cfg["cache_dir"],
        n_frames=int(data_cfg["n_frames"]),
        margin=float(data_cfg["relabel_margin"]),
        cache_frames=False,
    )
    if checkpoint is not None and checkpoint.get("feature_mean") is not None:
        dataset.set_feature_normalization(checkpoint["feature_mean"], checkpoint["feature_std"])
    return dataset


def build_model(config: dict[str, Any]) -> VideoSelector:
    data_cfg = config["data"]
    model_cfg = config["model"]
    return VideoSelector(
        n_frames=int(data_cfg["n_frames"]),
        n_classes=int(model_cfg["n_classes"]),
        input_dim=int(model_cfg.get("input_dim", 25)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        aggregation=str(model_cfg["aggregation"]),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )


def evaluate_model(
    model: VideoSelector,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="evaluate", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            predictions = logits.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_targets.extend(targets.numpy().tolist())

    return np.asarray(all_targets), np.asarray(all_predictions)


def format_confusion_matrix(conf_mat: np.ndarray) -> str:
    labels = [INDEX_TO_CHANNEL[idx] for idx in range(conf_mat.shape[0])]
    header = f"{'true/pred':<12} | " + " | ".join(f"{label:>10}" for label in labels)
    sep = "-" * len(header)
    lines = [header, sep]
    for row_idx, row in enumerate(conf_mat):
        lines.append(f"{labels[row_idx]:<12} | " + " | ".join(f"{int(value):>10}" for value in row))
    return "\n".join(lines)


def compute_scheduler_metrics(
    dataset: VideoSelectorDataset,
    predictions: np.ndarray,
    compute_cost: dict[str, float],
) -> dict[str, Any]:
    if len(predictions) != len(dataset.samples):
        raise ValueError("Prediction count does not match dataset.samples length.")

    pred_f1_values: list[float] = []
    pred_cost_values: list[float] = []
    oracle_f1_values: list[float] = []
    oracle_cost_values: list[float] = []

    for sample, pred_idx in zip(dataset.samples, predictions):
        pred_channel = INDEX_TO_CHANNEL[int(pred_idx)]
        pred_f1_values.append(float(sample.channel_f1[pred_channel]))
        pred_cost_values.append(float(compute_cost[pred_channel]))

        best_f1 = max(sample.channel_f1.values())
        tied_channels = [
            name for name in ("kalman", "gmc", "inference") if abs(sample.channel_f1[name] - best_f1) < 1e-9
        ]
        oracle_channel = tied_channels[0]
        oracle_f1_values.append(float(best_f1))
        oracle_cost_values.append(float(compute_cost[oracle_channel]))

    baseline: dict[str, dict[str, float]] = {}
    for channel_name in ("kalman", "gmc", "inference"):
        avg_f1 = float(np.mean([float(sample.channel_f1[channel_name]) for sample in dataset.samples]))
        avg_compute = float(compute_cost[channel_name])
        baseline[channel_name] = {
            "avg_f1": avg_f1,
            "compute": avg_compute,
            "f1_per_compute": avg_f1 / max(avg_compute, 1e-12),
        }

    selector_avg_f1 = float(np.mean(pred_f1_values))
    selector_avg_compute = float(np.mean(pred_cost_values))
    oracle_avg_f1 = float(np.mean(oracle_f1_values))
    oracle_avg_compute = float(np.mean(oracle_cost_values))

    return {
        "selector_avg_f1": selector_avg_f1,
        "selector_avg_compute": selector_avg_compute,
        "selector_f1_per_compute": selector_avg_f1 / max(selector_avg_compute, 1e-12),
        "oracle_avg_f1": oracle_avg_f1,
        "oracle_avg_compute": oracle_avg_compute,
        "oracle_f1_per_compute": oracle_avg_f1 / max(oracle_avg_compute, 1e-12),
        "baseline": baseline,
    }


def print_baseline_table(scheduler_metrics: dict[str, Any]) -> None:
    baseline = scheduler_metrics["baseline"]
    rows = [
        ("All-Kalman", baseline["kalman"]["avg_f1"], baseline["kalman"]["compute"], baseline["kalman"]["f1_per_compute"]),
        ("All-GMC", baseline["gmc"]["avg_f1"], baseline["gmc"]["compute"], baseline["gmc"]["f1_per_compute"]),
        ("All-Inference", baseline["inference"]["avg_f1"], baseline["inference"]["compute"], baseline["inference"]["f1_per_compute"]),
        (
            "Ours (Selector)",
            scheduler_metrics["selector_avg_f1"],
            scheduler_metrics["selector_avg_compute"],
            scheduler_metrics["selector_f1_per_compute"],
        ),
        (
            "Oracle (Best)",
            scheduler_metrics["oracle_avg_f1"],
            scheduler_metrics["oracle_avg_compute"],
            scheduler_metrics["oracle_f1_per_compute"],
        ),
    ]

    print("=" * 60)
    print(f"{'Method':<15} | {'Avg F1':>7} | {'Compute':>7} | {'F1/Compute':>10}")
    print("-" * 60)
    for name, avg_f1, compute, ratio in rows:
        print(f"{name:<15} | {avg_f1:>7.4f} | {compute:>7.3f} | {ratio:>10.3f}")
    print("=" * 60)


def save_results(
    results_dir: Path,
    checkpoint_path: Path,
    classification_metrics: dict[str, Any],
    scheduler_metrics: dict[str, Any],
    confusion: np.ndarray,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"eval_{checkpoint_path.stem}.json"
    payload = {
        "checkpoint": str(checkpoint_path),
        "classification": classification_metrics,
        "scheduler": scheduler_metrics,
        "confusion_matrix": confusion.tolist(),
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("evaluate")

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    val_dataset = create_val_dataset(config, checkpoint=checkpoint)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    targets, predictions = evaluate_model(model, val_loader, device)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    conf_mat = confusion_matrix(targets, predictions, labels=[0, 1, 2])
    acc = float(accuracy_score(targets, predictions))
    macro_f1 = float(np.mean(f1))

    logger.info("Part A: 分类指标")
    logger.info("整体 Acc=%.4f Macro-F1=%.4f", acc, macro_f1)
    logger.info("混淆矩阵:\n%s", format_confusion_matrix(conf_mat))
    for class_idx, class_name in INDEX_TO_CHANNEL.items():
        logger.info(
            "%s: precision=%.4f recall=%.4f f1=%.4f support=%d",
            class_name,
            precision[class_idx],
            recall[class_idx],
            f1[class_idx],
            int(support[class_idx]),
        )

    classification_metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
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

    compute_cost = dict(config["eval"]["compute_cost"])
    scheduler_metrics = compute_scheduler_metrics(val_dataset, predictions, compute_cost)
    logger.info("Part B: 调度指标")
    logger.info(
        "Selector: avg_f1=%.4f compute=%.3f f1/compute=%.3f",
        scheduler_metrics["selector_avg_f1"],
        scheduler_metrics["selector_avg_compute"],
        scheduler_metrics["selector_f1_per_compute"],
    )

    logger.info("Part C: Baseline 对比")
    print_baseline_table(scheduler_metrics)

    results_dir = Path(config["eval"].get("results_dir", "results"))
    output_path = save_results(
        results_dir=results_dir,
        checkpoint_path=checkpoint_path,
        classification_metrics=classification_metrics,
        scheduler_metrics=scheduler_metrics,
        confusion=conf_mat,
    )
    logger.info("评估结果已保存到: %s", output_path)


if __name__ == "__main__":
    main()

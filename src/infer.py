from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import INDEX_TO_CHANNEL, VideoSelectorDataset
from src.model import VideoSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the video selector model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Dataset split to run inference on.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many predictions to print in the console preview.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to results/infer_{checkpoint}_{split}.json.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Default 0 is more reliable on Windows.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to run inference on.",
    )
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


def create_dataset(
    config: dict[str, Any],
    split: str,
    checkpoint: dict[str, Any] | None = None,
) -> VideoSelectorDataset:
    data_cfg = config["data"]
    split_key = f"{split}_videos"
    video_paths, label_paths = resolve_paths(config, list(data_cfg[split_key]))
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


def run_inference(
    model: VideoSelector,
    dataloader: DataLoader,
    dataset: VideoSelectorDataset,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []
    sample_index = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="infer", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu()
            predictions = probs.argmax(dim=1)

            for row_idx in range(inputs.shape[0]):
                sample = dataset.get_sample(sample_index)
                pred_idx = int(predictions[row_idx].item())
                target_idx = int(targets[row_idx].item())
                prob_map = {
                    INDEX_TO_CHANNEL[class_idx]: float(probs[row_idx, class_idx].item())
                    for class_idx in range(probs.shape[1])
                }
                results.append(
                    {
                        "sample_index": sample_index,
                        "video_name": sample.video_name,
                        "frame_id": sample.frame_id,
                        "predicted_label": pred_idx,
                        "predicted_channel": INDEX_TO_CHANNEL[pred_idx],
                        "target_label": target_idx,
                        "target_channel": INDEX_TO_CHANNEL[target_idx],
                        "correct": pred_idx == target_idx,
                        "confidence": prob_map[INDEX_TO_CHANNEL[pred_idx]],
                        "probabilities": prob_map,
                        "best_channel_raw": sample.best_channel_raw,
                        "channel_f1": sample.channel_f1,
                    }
                )
                sample_index += 1

    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for item in results if item["correct"])
    prediction_counts = {name: 0 for name in INDEX_TO_CHANNEL.values()}
    target_counts = {name: 0 for name in INDEX_TO_CHANNEL.values()}

    for item in results:
        prediction_counts[item["predicted_channel"]] += 1
        target_counts[item["target_channel"]] += 1

    return {
        "num_samples": total,
        "num_correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "prediction_counts": prediction_counts,
        "target_counts": target_counts,
    }


def default_output_path(results_dir: str | Path, checkpoint_path: Path, split: str) -> Path:
    return Path(results_dir) / f"infer_{checkpoint_path.stem}_{split}.json"


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("infer")

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = create_dataset(config, args.split, checkpoint=checkpoint)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: max(0, int(args.max_samples))]
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Running inference on %s split with %d samples.", args.split, len(dataset))
    results = run_inference(model, dataloader, dataset, device)
    summary = summarize_results(results)

    logger.info(
        "Accuracy: %.4f (%d/%d)",
        summary["accuracy"],
        summary["num_correct"],
        summary["num_samples"],
    )
    logger.info("Prediction counts: %s", summary["prediction_counts"])
    logger.info("Target counts: %s", summary["target_counts"])

    preview_count = max(0, min(args.limit, len(results)))
    for item in results[:preview_count]:
        logger.info(
            "[%04d] %s frame=%d pred=%s target=%s conf=%.4f probs=%s",
            item["sample_index"],
            item["video_name"],
            item["frame_id"],
            item["predicted_channel"],
            item["target_channel"],
            item["confidence"],
            {key: round(value, 4) for key, value in item["probabilities"].items()},
        )

    output_path = Path(args.output) if args.output else default_output_path(
        config["eval"].get("results_dir", "results"),
        checkpoint_path,
        args.split,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "split": args.split,
                "summary": summary,
                "predictions": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved inference details to %s", output_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from src.dataset import INDEX_TO_CHANNEL, VideoSelectorDataset, load_label_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect selector-feature training labels.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_video_and_label_paths(
    config: dict[str, Any],
    video_names: list[str],
) -> tuple[list[Path], list[Path]]:
    video_dir = Path(config["data"].get("video_dir", ""))
    label_dir = Path(config["data"]["label_dir"])
    video_paths = [video_dir / f"{name}.mp4" for name in video_names]
    label_paths = [label_dir / f"{name}.json" for name in video_names]

    missing = [str(path) for path in label_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing label files:\n" + "\n".join(missing))

    return video_paths, label_paths


def format_distribution_table(rows: list[dict[str, int | str]]) -> str:
    header = f"{'Video':<24} | {'Total':>6} | {'Kalman':>6} | {'GMC':>6} | {'Inference':>9}"
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{str(row['video']):<24} | "
            f"{int(row['total']):>6} | "
            f"{int(row['kalman']):>6} | "
            f"{int(row['gmc']):>6} | "
            f"{int(row['inference']):>9}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("preprocess")

    config = load_config(args.config)
    data_cfg = config["data"]
    all_video_names = list(dict.fromkeys(data_cfg["train_videos"] + data_cfg["val_videos"]))
    video_paths, label_paths = resolve_video_and_label_paths(config, all_video_names)

    distribution_rows: list[dict[str, int | str]] = []
    total_counts = {0: 0, 1: 0, 2: 0}

    for video_path, label_path, video_name in zip(video_paths, label_paths, all_video_names):
        label_data = load_label_json(label_path)
        total_frames = int(label_data.get("summary", {}).get("total_frames", len(label_data.get("frames", []))))

        start_time = time.perf_counter()
        dataset = VideoSelectorDataset(
            video_paths=[video_path],
            label_paths=[label_path],
            cache_root=data_cfg["cache_dir"],
            n_frames=int(data_cfg["n_frames"]),
            margin=float(data_cfg["relabel_margin"]),
            cache_frames=False,
        )
        elapsed = time.perf_counter() - start_time

        labels = dataset.get_labels()
        video_counts = {
            0: sum(1 for label in labels if label == 0),
            1: sum(1 for label in labels if label == 1),
            2: sum(1 for label in labels if label == 2),
        }
        for key, value in video_counts.items():
            total_counts[key] += value

        distribution_rows.append(
            {
                "video": video_name,
                "total": len(labels),
                "kalman": video_counts[0],
                "gmc": video_counts[1],
                "inference": video_counts[2],
            }
        )

        logger.info(
            "Video %s processed: total_frames=%d, valid_samples=%d, elapsed=%.2fs",
            video_name,
            total_frames,
            len(labels),
            elapsed,
        )

    distribution_rows.append(
        {
            "video": "TOTAL",
            "total": sum(total_counts.values()),
            "kalman": total_counts[0],
            "gmc": total_counts[1],
            "inference": total_counts[2],
        }
    )

    logger.info("Relabeled class distribution:\n%s", format_distribution_table(distribution_rows))
    logger.info(
        "Class mapping: %s",
        ", ".join(f"{idx}={name}" for idx, name in INDEX_TO_CHANNEL.items()),
    )


if __name__ == "__main__":
    main()

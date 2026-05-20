from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


CHANNEL_TO_INDEX: dict[str, int] = {
    "kalman": 0,
    "gmc": 1,
    "inference": 2,
}

INDEX_TO_CHANNEL: dict[int, str] = {value: key for key, value in CHANNEL_TO_INDEX.items()}

SELECTOR_FEATURE_NAMES: tuple[str, ...] = (
    "frame_diff_mean",
    "frame_diff_std",
    "global_motion_dx",
    "global_motion_dy",
    "optical_flow_valid_ratio",
    "optical_flow_residual",
    "tracked_point_count",
    "is_bootstrap_frame",
    "last_gpu_box_count",
    "last_gpu_max_conf",
    "recent_gpu_box_count_mean",
    "recent_gpu_box_count_delta",
    "tracker_count",
    "confirmed_tracker_count",
    "mean_track_age",
    "mean_time_since_update",
    "max_time_since_update",
    "mean_speed",
    "mean_position_uncertainty",
    "max_position_uncertainty",
    "prediction_error_ma",
    "prediction_error_p95",
    "frames_since_last_gpu",
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoRecord:
    """Metadata for one frame-level selector JSON."""

    video_name: str
    label_path: Path


@dataclass(frozen=True)
class FrameSample:
    """One training sample corresponding to one target frame."""

    video_name: str
    frame_id: int
    label: int
    best_channel_raw: str
    feature_sequence: tuple[tuple[float, ...], ...]
    channel_f1: dict[str, float]


def relabel_with_margin(
    f1_kalman: float,
    f1_gmc: float,
    f1_inference: float,
    margin: float = 0.03,
) -> int:
    """Prefer cheaper channels when their F1 is close enough to the best F1."""

    scores = {
        "kalman": float(f1_kalman),
        "gmc": float(f1_gmc),
        "inference": float(f1_inference),
    }
    best_f1 = max(scores.values())
    for channel_name in ("kalman", "gmc", "inference"):
        if best_f1 - scores[channel_name] <= margin:
            return CHANNEL_TO_INDEX[channel_name]
    return CHANNEL_TO_INDEX["inference"]


def load_label_json(label_path: Path) -> dict[str, Any]:
    """Load one frame-level label JSON file."""

    with label_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_video_records(
    video_paths: list[str | Path] | None,
    label_paths: list[str | Path],
    cache_root: str | Path | None = None,
) -> list[VideoRecord]:
    """Build records from label paths.

    ``video_paths`` and ``cache_root`` are accepted for compatibility with the
    previous image dataset, but selector-feature training does not use them.
    """

    if video_paths is not None and len(video_paths) != len(label_paths):
        raise ValueError("video_paths and label_paths must have the same length.")

    records: list[VideoRecord] = []
    for label_path in label_paths:
        label_path = Path(label_path)
        records.append(VideoRecord(video_name=label_path.stem, label_path=label_path))
    return records


class VideoSelectorDataset(Dataset[tuple[Tensor, int]]):
    """Selector feature-sequence dataset.

    Each sample returns:
    - x: Tensor of shape [n_frames, feature_dim]
    - y: int label in {0, 1, 2}

    Missing history is padded by repeating the earliest available frame.
    """

    def __init__(
        self,
        video_paths: list[str | Path] | None,
        label_paths: list[str | Path],
        cache_root: str | Path = "cache",
        n_frames: int = 4,
        image_size: int | None = None,
        margin: float = 0.03,
        cache_frames: bool = False,
        feature_names: tuple[str, ...] = SELECTOR_FEATURE_NAMES,
        feature_mean: list[float] | Tensor | None = None,
        feature_std: list[float] | Tensor | None = None,
        **_: Any,
    ) -> None:
        if n_frames <= 0:
            raise ValueError("n_frames must be positive.")

        self.records = build_video_records(video_paths, label_paths, cache_root)
        self.n_frames = n_frames
        self.margin = margin
        self.feature_names = tuple(feature_names)
        self.feature_dim = len(self.feature_names)

        self.total_frames = 0
        self.valid_frames = 0
        self.skipped_missing_channel = 0
        self.skipped_missing_features = 0
        self.missing_channel_counts = {channel: 0 for channel in CHANNEL_TO_INDEX}
        self.samples = self._build_samples()

        self.feature_mean: Tensor | None = None
        self.feature_std: Tensor | None = None
        if feature_mean is not None and feature_std is not None:
            self.set_feature_normalization(feature_mean, feature_std)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        x = torch.tensor(sample.feature_sequence, dtype=torch.float32)
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std
        return x, sample.label

    def prepare_cache(self) -> None:
        """Compatibility no-op: frames are no longer cached or loaded."""

    def get_sample(self, index: int) -> FrameSample:
        return self.samples[index]

    def get_labels(self) -> list[int]:
        return [sample.label for sample in self.samples]

    def get_feature_names(self) -> tuple[str, ...]:
        return self.feature_names

    @staticmethod
    def count_cached_frames(cache_dir: str | Path) -> int:
        return 0

    def compute_feature_normalization(self) -> tuple[Tensor, Tensor]:
        if not self.samples:
            raise ValueError("Cannot compute feature normalization on an empty dataset.")
        values = np.asarray(
            [frame for sample in self.samples for frame in sample.feature_sequence],
            dtype=np.float32,
        )
        mean = torch.tensor(values.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(values.std(axis=0), dtype=torch.float32)
        std = torch.clamp(std, min=1e-6)
        return mean, std

    def set_feature_normalization(
        self,
        feature_mean: list[float] | Tensor,
        feature_std: list[float] | Tensor,
    ) -> None:
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.numel() != self.feature_dim or std.numel() != self.feature_dim:
            raise ValueError(
                f"Expected normalization length {self.feature_dim}, "
                f"got mean={mean.numel()} std={std.numel()}"
            )
        self.feature_mean = mean
        self.feature_std = torch.clamp(std, min=1e-6)

    def _build_samples(self) -> list[FrameSample]:
        samples: list[FrameSample] = []
        for record in self.records:
            label_data = load_label_json(record.label_path)
            frames = label_data.get("frames", [])
            frame_features: list[tuple[float, ...] | None] = []

            for frame_info in frames:
                features = frame_info.get("selector_features")
                if isinstance(features, dict):
                    frame_features.append(self._feature_vector(features))
                else:
                    frame_features.append(None)

            for frame_pos, frame_info in enumerate(frames):
                self.total_frames += 1
                channel_f1 = self._extract_channel_f1(frame_info)
                if channel_f1 is None:
                    self.skipped_missing_channel += 1
                    continue

                feature_sequence = self._history_features(frame_features, frame_pos)
                if feature_sequence is None:
                    self.skipped_missing_features += 1
                    continue

                frame_id = int(frame_info["frame_id"])
                label = relabel_with_margin(
                    f1_kalman=channel_f1["kalman"],
                    f1_gmc=channel_f1["gmc"],
                    f1_inference=channel_f1["inference"],
                    margin=self.margin,
                )
                samples.append(
                    FrameSample(
                        video_name=record.video_name,
                        frame_id=frame_id,
                        label=label,
                        best_channel_raw=str(frame_info.get("best_channel", "")),
                        feature_sequence=feature_sequence,
                        channel_f1=channel_f1,
                    )
                )
                self.valid_frames += 1

        self._log_dataset_build_summary()
        return samples

    def _feature_vector(self, features: dict[str, Any]) -> tuple[float, ...]:
        return tuple(float(features.get(name, 0.0) or 0.0) for name in self.feature_names)

    def _history_features(
        self,
        frame_features: list[tuple[float, ...] | None],
        frame_pos: int,
    ) -> tuple[tuple[float, ...], ...] | None:
        first_valid = next((item for item in frame_features[: frame_pos + 1] if item is not None), None)
        if first_valid is None:
            return None

        history: list[tuple[float, ...]] = []
        for pos in range(frame_pos - self.n_frames + 1, frame_pos + 1):
            if pos < 0:
                history.append(first_valid)
                continue
            features = frame_features[pos]
            if features is None:
                return None
            history.append(features)
        return tuple(history)

    def _extract_channel_f1(self, frame_info: dict[str, Any]) -> dict[str, float] | None:
        channels = frame_info.get("channels", {})
        scores: dict[str, float] = {}
        missing_any = False
        for channel_name in ("kalman", "gmc", "inference"):
            channel_info = channels.get(channel_name)
            metrics = channel_info.get("metrics") if isinstance(channel_info, dict) else None
            f1_value = metrics.get("f1") if isinstance(metrics, dict) else None
            if f1_value is None:
                self.missing_channel_counts[channel_name] += 1
                missing_any = True
                continue
            scores[channel_name] = float(f1_value)
        if missing_any:
            return None
        return scores

    def _log_dataset_build_summary(self) -> None:
        LOGGER.info(
            "Total frames: %d, valid samples: %d, skipped missing channel: %d, "
            "skipped missing selector_features: %d",
            self.total_frames,
            self.valid_frames,
            self.skipped_missing_channel,
            self.skipped_missing_features,
        )
        if self.total_frames <= 0:
            return
        for channel_name, missing_count in self.missing_channel_counts.items():
            missing_ratio = missing_count / self.total_frames
            if missing_ratio >= 0.1:
                LOGGER.warning(
                    "Channel %s has many missing metrics: %d / %d (%.2f%%)",
                    channel_name,
                    missing_count,
                    self.total_frames,
                    missing_ratio * 100.0,
                )


def discover_video_label_pairs(
    videos_dir: str | Path,
    labels_dir: str | Path,
    video_suffix: str = ".mp4",
    label_suffix: str = ".json",
) -> tuple[list[Path], list[Path]]:
    """Match labels by stem; video paths are returned only for compatibility."""

    videos_dir = Path(videos_dir)
    labels_dir = Path(labels_dir)
    label_paths = sorted(labels_dir.glob(f"*{label_suffix}"))
    if not label_paths:
        raise FileNotFoundError(f"No label files found in {labels_dir}.")

    video_paths = [videos_dir / f"{path.stem}{video_suffix}" for path in label_paths]
    return video_paths, label_paths


def compute_class_weights(labels: list[int]) -> Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""

    if not labels:
        raise ValueError("labels must not be empty.")

    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(CHANNEL_TO_INDEX))
    counts = np.maximum(counts, 1)
    total = float(counts.sum())
    num_classes = float(len(CHANNEL_TO_INDEX))
    weights = total / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)

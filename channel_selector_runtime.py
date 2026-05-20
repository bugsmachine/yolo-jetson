#!/usr/bin/env python3
"""Runtime channel selector for the Jetson pipeline.

This module loads the trained structured-feature GRU/LSTM selector and keeps
the short temporal window required by the model. It intentionally has no
dependency on raw images or detector internals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from channel_selector_interfaces import ChannelSelectorInput


CHANNEL_TO_ACTION = {
    "kalman": 0,
    "gmc": 1,
    "inference": 2,
}
ACTION_TO_CHANNEL = {value: key for key, value in CHANNEL_TO_ACTION.items()}

DEFAULT_FEATURE_NAMES = (
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


@dataclass
class ChannelSelectorDecision:
    action: int
    score: float
    channel: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    forced_reason: str = ""
    model_available: bool = False


class _VideoSelectorModel:
    """Small wrapper that builds the same architecture as training src/model.py."""

    def __init__(
        self,
        n_frames: int,
        n_classes: int,
        input_dim: int,
        hidden_dim: int,
        aggregation: str,
        dropout: float,
    ) -> None:
        import torch
        from torch import nn

        class VideoSelector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_frames = n_frames
                self.input_dim = input_dim
                self.aggregation = aggregation
                self.feature_encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
                if aggregation == "lstm":
                    self.temporal = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
                    head_in_dim = hidden_dim
                elif aggregation == "gru":
                    self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
                    head_in_dim = hidden_dim
                else:
                    self.temporal = None
                    if aggregation == "mean":
                        head_in_dim = hidden_dim
                    elif aggregation == "concat":
                        head_in_dim = hidden_dim * n_frames
                    elif aggregation == "diff":
                        head_in_dim = hidden_dim * 2
                    else:
                        raise ValueError(f"Unsupported aggregation: {aggregation}")

                self.classifier = nn.Sequential(
                    nn.Linear(head_in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, n_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                frame_features = self.feature_encoder(x)
                if self.aggregation == "mean":
                    aggregated = frame_features.mean(dim=1)
                elif self.aggregation == "concat":
                    aggregated = frame_features.reshape(frame_features.shape[0], -1)
                elif self.aggregation == "diff":
                    current = frame_features[:, -1, :]
                    if frame_features.shape[1] == 1:
                        mean_diff = torch.zeros_like(current)
                    else:
                        mean_diff = (frame_features[:, 1:, :] - frame_features[:, :-1, :]).mean(dim=1)
                    aggregated = torch.cat([current, mean_diff], dim=1)
                elif self.aggregation == "lstm":
                    _, (hidden_state, _) = self.temporal(frame_features)
                    aggregated = hidden_state[-1]
                elif self.aggregation == "gru":
                    _, hidden_state = self.temporal(frame_features)
                    aggregated = hidden_state[-1]
                else:
                    raise RuntimeError(f"Unknown aggregation: {self.aggregation}")
                return self.classifier(aggregated)

        self.model = VideoSelector()


class RuntimeChannelSelector:
    """Stateful selector used by both multiprocessing and single-thread paths."""

    def __init__(
        self,
        checkpoint_path: str = "../checkpoints/best.pth",
        device: str = "cpu",
        fallback_low: float = 0.3,
        fallback_high: float = 0.7,
        max_skip_frames: int = 60,
        force_gpu_interval: int = 60,
        uncertainty_threshold: float = 100000.0,
        prediction_error_threshold: float = 0.9,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        self.fallback_low = float(fallback_low)
        self.fallback_high = float(fallback_high)
        self.max_skip_frames = int(max_skip_frames)
        self.force_gpu_interval = int(force_gpu_interval)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.prediction_error_threshold = float(prediction_error_threshold)

        self.feature_names: List[str] = list(DEFAULT_FEATURE_NAMES)
        self.n_frames = 60
        self.history: Deque[List[float]] = deque(maxlen=self.n_frames)
        self.actions: Deque[int] = deque(maxlen=max(self.max_skip_frames, self.force_gpu_interval, 1))
        self.model: Any = None
        self.torch: Any = None
        self.device: Any = None
        self.feature_mean: Any = None
        self.feature_std: Any = None
        self.model_available = False
        self.load_error = ""

        self._load_model()

    def decide(self, selector_input: ChannelSelectorInput) -> ChannelSelectorDecision:
        features = selector_input.to_feature_dict()
        vector = self._feature_vector(features)
        self.history.append(vector)

        forced_reason = self._guardrail_reason(selector_input)
        if forced_reason:
            decision = ChannelSelectorDecision(
                action=CHANNEL_TO_ACTION["inference"],
                score=1.0,
                channel="inference",
                forced_reason=forced_reason,
                model_available=self.model_available,
            )
            self.actions.append(decision.action)
            return decision

        if self.model_available:
            decision = self._model_decision()
        else:
            decision = self._fallback_decision(selector_input)

        self.actions.append(decision.action)
        return decision

    def _load_model(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.load_error = f"torch unavailable: {exc}"
            return

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            config = checkpoint.get("config", {})
            data_cfg = config.get("data", {})
            model_cfg = config.get("model", {})
            self.n_frames = int(data_cfg.get("n_frames", self.n_frames))
            self.feature_names = list(checkpoint.get("feature_names") or self.feature_names)
            input_dim = int(model_cfg.get("input_dim", len(self.feature_names)))
            if input_dim != len(self.feature_names):
                raise ValueError(
                    f"checkpoint input_dim={input_dim}, feature_names={len(self.feature_names)}"
                )

            self.history = deque(maxlen=self.n_frames)
            self.device = torch.device(self.device_name if self.device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
            wrapper = _VideoSelectorModel(
                n_frames=self.n_frames,
                n_classes=int(model_cfg.get("n_classes", 3)),
                input_dim=input_dim,
                hidden_dim=int(model_cfg.get("hidden_dim", 128)),
                aggregation=str(model_cfg.get("aggregation", "gru")),
                dropout=float(model_cfg.get("dropout", 0.0)),
            )
            self.model = wrapper.model.to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            if checkpoint.get("feature_mean") is not None and checkpoint.get("feature_std") is not None:
                self.feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=self.device)
                self.feature_std = torch.clamp(
                    torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32, device=self.device),
                    min=1e-6,
                )
            self.torch = torch
            self.model_available = True
        except Exception as exc:
            self.load_error = str(exc)
            self.model_available = False
            self.model = None

    def _feature_vector(self, features: Dict[str, float]) -> List[float]:
        return [float(features.get(name, 0.0) or 0.0) for name in self.feature_names]

    def _model_decision(self) -> ChannelSelectorDecision:
        assert self.torch is not None
        assert self.model is not None

        frames = list(self.history)
        if not frames:
            frames = [[0.0] * len(self.feature_names)]
        while len(frames) < self.n_frames:
            frames.insert(0, frames[0])

        x = self.torch.tensor([frames[-self.n_frames:]], dtype=self.torch.float32, device=self.device)
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std

        with self.torch.no_grad():
            logits = self.model(x)
            probs_tensor = self.torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs_tensor))
        channel = ACTION_TO_CHANNEL.get(pred_idx, "inference")
        probabilities = {
            ACTION_TO_CHANNEL[idx]: float(probs_tensor[idx])
            for idx in range(min(len(probs_tensor), len(ACTION_TO_CHANNEL)))
        }

        # 模型只选 kalman vs gmc；YOLO 校准由 guardrail (max_skip_frames) 控制
        if channel == "inference":
            kalman_prob = probabilities.get("kalman", 0.0)
            gmc_prob = probabilities.get("gmc", 0.0)
            channel = "kalman" if kalman_prob >= gmc_prob else "gmc"
            pred_idx = CHANNEL_TO_ACTION[channel]

        return ChannelSelectorDecision(
            action=pred_idx,
            score=float(probs_tensor[pred_idx]),
            channel=channel,
            probabilities=probabilities,
            model_available=True,
        )

    def _fallback_decision(self, selector_input: ChannelSelectorInput) -> ChannelSelectorDecision:
        score = float(selector_input.frame_features.frame_diff_mean)
        # 无模型时只在 kalman/gmc 间选，不主动触发 YOLO
        channel = "kalman" if score < self.fallback_low else "gmc"
        return ChannelSelectorDecision(
            action=CHANNEL_TO_ACTION[channel],
            score=score,
            channel=channel,
            model_available=False,
        )

    def _guardrail_reason(self, selector_input: ChannelSelectorInput) -> str:
        consecutive_skips = 0
        for action in reversed(self.actions):
            if action == CHANNEL_TO_ACTION["inference"]:
                break
            consecutive_skips += 1
        if consecutive_skips >= self.max_skip_frames:
            return "max_skip_frames"

        return ""

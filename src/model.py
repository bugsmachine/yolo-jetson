from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


AggregationType = Literal["mean", "concat", "diff", "lstm", "gru"]


class VideoSelector(nn.Module):
    """Channel selector for structured per-frame features.

    The input is a sequence of selector feature vectors with shape [B, T, F],
    not raw images. The class name is kept for compatibility with the existing
    training and evaluation scripts.
    """

    def __init__(
        self,
        n_frames: int = 4,
        n_classes: int = 3,
        input_dim: int = 25,
        hidden_dim: int = 128,
        aggregation: AggregationType = "gru",
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if n_frames <= 0:
            raise ValueError("n_frames must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if aggregation not in {"mean", "concat", "diff", "lstm", "gru"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        self.n_frames = n_frames
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        if aggregation == "lstm":
            self.temporal: nn.Module | None = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            head_in_dim = hidden_dim
        elif aggregation == "gru":
            self.temporal = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            head_in_dim = hidden_dim
        else:
            self.temporal = None
            if aggregation == "mean":
                head_in_dim = hidden_dim
            elif aggregation == "concat":
                head_in_dim = hidden_dim * n_frames
            else:
                head_in_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run classification.

        Args:
            x: Tensor of shape [B, T, F].
        Returns:
            Logits of shape [B, n_classes].
        """

        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, T, F], got {tuple(x.shape)}")
        if x.shape[1] != self.n_frames:
            raise ValueError(f"Expected T={self.n_frames}, got T={x.shape[1]}")
        if x.shape[2] != self.input_dim:
            raise ValueError(f"Expected F={self.input_dim}, got F={x.shape[2]}")

        frame_features = self.feature_encoder(x)
        aggregated = self.aggregate_features(frame_features)
        return self.classifier(aggregated)

    def aggregate_features(self, frame_features: Tensor) -> Tensor:
        """Aggregate encoded frame features into one sequence feature."""

        if self.aggregation == "mean":
            return frame_features.mean(dim=1)

        if self.aggregation == "concat":
            return frame_features.reshape(frame_features.shape[0], -1)

        if self.aggregation == "diff":
            current_feature = frame_features[:, -1, :]
            if frame_features.shape[1] == 1:
                mean_diff = torch.zeros_like(current_feature)
            else:
                diffs = frame_features[:, 1:, :] - frame_features[:, :-1, :]
                mean_diff = diffs.mean(dim=1)
            return torch.cat([current_feature, mean_diff], dim=1)

        if self.aggregation == "lstm":
            if not isinstance(self.temporal, nn.LSTM):
                raise RuntimeError("LSTM aggregator requested but temporal module is missing.")
            _, (hidden_state, _) = self.temporal(frame_features)
            return hidden_state[-1]

        if self.aggregation == "gru":
            if not isinstance(self.temporal, nn.GRU):
                raise RuntimeError("GRU aggregator requested but temporal module is missing.")
            _, hidden_state = self.temporal(frame_features)
            return hidden_state[-1]

        raise RuntimeError(f"Unknown aggregation type: {self.aggregation}")

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Compatibility no-op: this feature model has no image backbone."""

    def backbone_parameters(self):
        """Compatibility helper for old two-phase training code."""

        return iter(())

    def classifier_parameters(self):
        """Return all trainable parameters."""

        return self.parameters()

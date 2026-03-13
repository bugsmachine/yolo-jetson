#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class FrameFeatureSnapshot:
    frame_id: int
    frame_diff_mean: float = 0.0
    frame_diff_std: float = 0.0
    global_motion_dx: float = 0.0
    global_motion_dy: float = 0.0
    optical_flow_valid_ratio: float = 0.0
    optical_flow_residual: float = 0.0
    tracked_point_count: int = 0
    is_bootstrap_frame: bool = False

    def to_dict(self) -> Dict[str, float]:
        return {
            "frame_id": float(self.frame_id),
            "frame_diff_mean": self.frame_diff_mean,
            "frame_diff_std": self.frame_diff_std,
            "global_motion_dx": self.global_motion_dx,
            "global_motion_dy": self.global_motion_dy,
            "optical_flow_valid_ratio": self.optical_flow_valid_ratio,
            "optical_flow_residual": self.optical_flow_residual,
            "tracked_point_count": float(self.tracked_point_count),
            "is_bootstrap_frame": float(self.is_bootstrap_frame),
        }


@dataclass
class StateSnapshot:
    frame_id: int = -1
    last_gpu_frame_id: int = -1
    last_gpu_source: str = ""
    last_gpu_box_count: int = 0
    last_gpu_max_conf: float = 0.0
    recent_gpu_box_count_mean: float = 0.0
    recent_gpu_box_count_delta: float = 0.0
    tracker_count: int = 0
    confirmed_tracker_count: int = 0
    mean_track_age: float = 0.0
    mean_time_since_update: float = 0.0
    max_time_since_update: int = 0
    mean_speed: float = 0.0
    mean_position_uncertainty: float = 0.0
    max_position_uncertainty: float = 0.0
    prediction_error_ma: float = 0.0
    prediction_error_p95: float = 0.0

    def frames_since_last_gpu(self, current_frame_id: int) -> int:
        if self.last_gpu_frame_id < 0:
            return -1
        return max(0, current_frame_id - self.last_gpu_frame_id)

    def to_dict(self, current_frame_id: Optional[int] = None) -> Dict[str, float]:
        frames_since_last_gpu = (
            self.frames_since_last_gpu(current_frame_id)
            if current_frame_id is not None
            else -1
        )
        return {
            "frame_id": float(self.frame_id),
            "last_gpu_frame_id": float(self.last_gpu_frame_id),
            "last_gpu_box_count": float(self.last_gpu_box_count),
            "last_gpu_max_conf": self.last_gpu_max_conf,
            "recent_gpu_box_count_mean": self.recent_gpu_box_count_mean,
            "recent_gpu_box_count_delta": self.recent_gpu_box_count_delta,
            "tracker_count": float(self.tracker_count),
            "confirmed_tracker_count": float(self.confirmed_tracker_count),
            "mean_track_age": self.mean_track_age,
            "mean_time_since_update": self.mean_time_since_update,
            "max_time_since_update": float(self.max_time_since_update),
            "mean_speed": self.mean_speed,
            "mean_position_uncertainty": self.mean_position_uncertainty,
            "max_position_uncertainty": self.max_position_uncertainty,
            "prediction_error_ma": self.prediction_error_ma,
            "prediction_error_p95": self.prediction_error_p95,
            "frames_since_last_gpu": float(frames_since_last_gpu),
        }


@dataclass
class ChannelSelectorInput:
    frame_features: FrameFeatureSnapshot
    state_snapshot: StateSnapshot = field(default_factory=StateSnapshot)

    def to_feature_dict(self) -> Dict[str, float]:
        features = self.frame_features.to_dict()
        features.update(self.state_snapshot.to_dict(self.frame_features.frame_id))
        return features

    def feature_names(self) -> List[str]:
        return list(self.to_feature_dict().keys())


class ReaderFeatureExtractor:
    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
        )

    def extract(self, frame: np.ndarray, frame_id: int) -> FrameFeatureSnapshot:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            tracked_point_count = len(self.prev_points) if self.prev_points is not None else 0
            return FrameFeatureSnapshot(
                frame_id=frame_id,
                frame_diff_mean=1.0,
                frame_diff_std=0.0,
                tracked_point_count=tracked_point_count,
                is_bootstrap_frame=True,
            )

        diff = cv2.absdiff(self.prev_gray, gray)
        frame_diff_mean = float(np.mean(diff) / 255.0)
        frame_diff_std = float(np.std(diff) / 255.0)
        prev_points = self.prev_points

        if prev_points is None or len(prev_points) < 10:
            prev_points = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=None, **self.feature_params
            )

        dx = 0.0
        dy = 0.0
        valid_ratio = 0.0
        residual = 0.0
        tracked_point_count = 0

        if prev_points is not None and len(prev_points) > 0:
            next_points, status, _err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_points, None, **self.lk_params
            )
            if next_points is not None and status is not None:
                status = status.flatten()
                good_old = prev_points[status == 1].reshape(-1, 2)
                good_new = next_points[status == 1].reshape(-1, 2)
                tracked_point_count = len(good_new)
                valid_ratio = tracked_point_count / max(len(prev_points), 1)

                if tracked_point_count >= 5:
                    motion_vectors = good_new - good_old
                    median_motion = np.median(motion_vectors, axis=0)
                    dx = float(median_motion[0])
                    dy = float(median_motion[1])
                    residuals = np.linalg.norm(motion_vectors - median_motion, axis=1)
                    residual = float(np.median(residuals)) if len(residuals) > 0 else 0.0

                    if tracked_point_count < 30:
                        self.prev_points = cv2.goodFeaturesToTrack(
                            gray, mask=None, **self.feature_params
                        )
                    else:
                        self.prev_points = good_new.reshape(-1, 1, 2)
                else:
                    self.prev_points = cv2.goodFeaturesToTrack(
                        gray, mask=None, **self.feature_params
                    )
            else:
                self.prev_points = cv2.goodFeaturesToTrack(
                    gray, mask=None, **self.feature_params
                )
        else:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )

        self.prev_gray = gray

        return FrameFeatureSnapshot(
            frame_id=frame_id,
            frame_diff_mean=frame_diff_mean,
            frame_diff_std=frame_diff_std,
            global_motion_dx=dx,
            global_motion_dy=dy,
            optical_flow_valid_ratio=float(valid_ratio),
            optical_flow_residual=residual,
            tracked_point_count=tracked_point_count,
            is_bootstrap_frame=False,
        )

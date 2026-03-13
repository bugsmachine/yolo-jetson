#!/usr/bin/env python3
"""
多目标卡尔曼滤波跟踪器
实现基于卡尔曼滤波的目标状态预测与更新


设计思路：
1. 每个被跟踪目标有独立的KalmanTracker
2. 使用匀速运动模型预测下一帧位置
3. 检测结果与跟踪器匹配后更新状态
4. 长时间未匹配的跟踪器删除
"""


import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class KalmanTrackerConfig:
    """卡尔曼跟踪器配置"""
    # 过程噪声系数（控制对运动变化的适应性）
    process_noise_scale: float = 1.0
    # 测量噪声系数（控制对检测结果的信任度）
    measurement_noise_scale: float = 1.0
    # 初始速度不确定性
    initial_velocity_std: float = 10.0
    # 最大丢失帧数（超过则删除跟踪器）
    max_age: int = 30
    # 最小命中次数（低于则认为不稳定）
    min_hits: int = 3
    # IoU匹配阈值
    iou_threshold: float = 0.3

class SingleObjectKalmanTracker:
    """
    单目标卡尔曼跟踪器

    状态向量: [x_center, y_center, area, aspect_ratio, vx, vy, va, vr]
    其中:
        - x_center, y_center: 框中心坐标
        - area: 框面积
        - aspect_ratio: 宽高比 (w/h)
        - vx, vy, va, vr: 对应的速度

    使用匀速运动模型(Constant Velocity Model)
    """

    _count = 0

    def __init__(self, bbox: Dict, config: KalmanTrackerConfig = None):
        """
        初始化跟踪器

        Args:
            bbox: 检测框 {'x1', 'y1', 'x2', 'y2', 'conf', 'class'}
            config: 配置参数
        """
        self.config = config or KalmanTrackerConfig()

        SingleObjectKalmanTracker._count += 1
        self.id = SingleObjectKalmanTracker._count

        self.state = self._bbox_to_state(bbox)

        self.dim_x = 8
        self.dim_z = 4

        self.x = np.zeros((self.dim_x, 1))
        self.x[:4, 0] = self.state

        self.P = np.eye(self.dim_x)

        self.P[4:, 4:] *= self.config.initial_velocity_std ** 2

        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1
        self.F[3, 7] = 1

        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(self.dim_x) * self.config.process_noise_scale
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(self.dim_z) * self.config.measurement_noise_scale

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        self.confidence = bbox.get('conf', 0.0)
        self.class_id = bbox.get('class', 0)

        self.history: List[Dict] = []

    @staticmethod
    def _bbox_to_state(bbox: Dict) -> np.ndarray:
        """将bbox [x1,y1,x2,y2] 转换为状态 [x_center, y_center, area, aspect_ratio]"""
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        w = x2 - x1
        h = y2 - y1
        x_center = x1 + w / 2
        y_center = y1 + h / 2
        area = w * h
        aspect_ratio = w / max(h, 1e-6)
        return np.array([x_center, y_center, area, aspect_ratio])

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> Dict:
        """将状态 [x_center, y_center, area, aspect_ratio] 转换为bbox [x1,y1,x2,y2]"""
        x_center, y_center, area, aspect_ratio = state[:4]

        area = max(area, 1.0)
        aspect_ratio = max(aspect_ratio, 0.1)

        h = np.sqrt(area / aspect_ratio)
        w = aspect_ratio * h

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        return {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}

    def predict(self) -> Dict:
        """
        预测下一帧位置

        Returns:
            预测的bbox
        """

        self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1

        predicted_bbox = self._state_to_bbox(self.x.flatten())
        predicted_bbox['conf'] = self.confidence * 0.95
        predicted_bbox['class'] = self.class_id
        predicted_bbox['track_id'] = self.id

        self.history.append(predicted_bbox.copy())

        return predicted_bbox

    def update(self, bbox: Dict):
        """
        用检测结果更新状态

        Args:
            bbox: 检测框
        """

        z = self._bbox_to_state(bbox).reshape(-1, 1)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y = z - self.H @ self.x
        self.x = self.x + K @ y

        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0
        self.confidence = bbox.get('conf', self.confidence)
        self.class_id = bbox.get('class', self.class_id)

    def get_current_bbox(self) -> Dict:
        """获取当前状态对应的bbox"""
        bbox = self._state_to_bbox(self.x.flatten())
        bbox['conf'] = self.confidence
        bbox['class'] = self.class_id
        bbox['track_id'] = self.id
        return bbox

    def get_speed(self) -> float:
        vx = float(self.x[4, 0])
        vy = float(self.x[5, 0])
        return float(np.hypot(vx, vy))

    def get_position_uncertainty(self) -> float:
        return float(np.trace(self.P[:4, :4]) / 4.0)

    def is_confirmed(self) -> bool:
        """是否为确认的跟踪器（命中次数足够）"""
        return self.hits >= self.config.min_hits

    def is_dead(self) -> bool:
        """是否应该删除（丢失时间过长）"""
        return self.time_since_update > self.config.max_age

class MultiObjectKalmanTracker:
    """
    多目标卡尔曼跟踪器

    管理多个SingleObjectKalmanTracker，负责：
    1. 检测结果与跟踪器的匹配
    2. 新目标的创建
    3. 丢失目标的删除
    4. 预测所有目标的下一帧位置
    """

    def __init__(self, config: KalmanTrackerConfig = None):
        self.config = config or KalmanTrackerConfig()
        self.trackers: List[SingleObjectKalmanTracker] = []
        self.frame_count = 0

    @staticmethod
    def compute_iou(box1: Dict, box2: Dict) -> float:
        """计算两个框的IoU"""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

        union_area = box1_area + box2_area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _match_detections_to_trackers(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        将检测结果与现有跟踪器匹配

        Args:
            detections: 检测结果列表

        Returns:
            matches: [(detection_idx, tracker_idx), ...]
            unmatched_detections: [detection_idx, ...]
            unmatched_trackers: [tracker_idx, ...]
        """
        if not self.trackers or not detections:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        num_det = len(detections)
        num_trk = len(self.trackers)
        iou_matrix = np.zeros((num_det, num_trk))

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                trk_bbox = trk.get_current_bbox()
                iou_matrix[d, t] = self.compute_iou(det, trk_bbox)

        matches = []
        matched_det = set()
        matched_trk = set()

        pairs = []
        for d in range(num_det):
            for t in range(num_trk):
                if iou_matrix[d, t] >= self.config.iou_threshold:
                    pairs.append((iou_matrix[d, t], d, t))

        pairs.sort(reverse=True)

        for iou, d, t in pairs:
            if d not in matched_det and t not in matched_trk:
                matches.append((d, t))
                matched_det.add(d)
                matched_trk.add(t)

        unmatched_detections = [d for d in range(num_det) if d not in matched_det]
        unmatched_trackers = [t for t in range(num_trk) if t not in matched_trk]

        return matches, unmatched_detections, unmatched_trackers

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        用检测结果更新跟踪器

        Args:
            detections: 检测结果列表

        Returns:
            当前帧的跟踪结果（带track_id）
        """
        self.frame_count += 1

        for trk in self.trackers:
            trk.predict()

        matches, unmatched_dets, unmatched_trks = self._match_detections_to_trackers(detections)

        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_dets:
            new_tracker = SingleObjectKalmanTracker(detections[det_idx], self.config)
            self.trackers.append(new_tracker)

        self.trackers = [trk for trk in self.trackers if not trk.is_dead()]

        results = []
        for trk in self.trackers:
            if trk.is_confirmed() or trk.time_since_update == 0:
                results.append(trk.get_current_bbox())

        return results

    def predict(self) -> List[Dict]:
        """
        预测下一帧位置（不使用检测结果更新）
        用于早退场景（跳过推理）

        Returns:
            预测的目标列表
        """
        self.frame_count += 1

        results = []
        for trk in self.trackers:
            if not trk.is_dead():
                predicted_bbox = trk.predict()

                if trk.is_confirmed():
                    results.append(predicted_bbox)

        self.trackers = [trk for trk in self.trackers if not trk.is_dead()]

        return results

    def get_tracker_count(self) -> int:
        """获取当前跟踪器数量"""
        return len(self.trackers)

    def get_confirmed_tracker_count(self) -> int:
        """获取确认的跟踪器数量"""
        return sum(1 for trk in self.trackers if trk.is_confirmed())

    def get_state_summary(self) -> Dict[str, float]:
        if not self.trackers:
            return {
                'tracker_count': 0,
                'confirmed_tracker_count': 0,
                'mean_track_age': 0.0,
                'mean_time_since_update': 0.0,
                'max_time_since_update': 0,
                'mean_speed': 0.0,
                'mean_position_uncertainty': 0.0,
                'max_position_uncertainty': 0.0,
            }

        ages = np.array([trk.age for trk in self.trackers], dtype=float)
        stale = np.array([trk.time_since_update for trk in self.trackers], dtype=float)
        speeds = np.array([trk.get_speed() for trk in self.trackers], dtype=float)
        uncertainty = np.array(
            [trk.get_position_uncertainty() for trk in self.trackers], dtype=float
        )

        return {
            'tracker_count': len(self.trackers),
            'confirmed_tracker_count': self.get_confirmed_tracker_count(),
            'mean_track_age': float(np.mean(ages)),
            'mean_time_since_update': float(np.mean(stale)),
            'max_time_since_update': int(np.max(stale)),
            'mean_speed': float(np.mean(speeds)),
            'mean_position_uncertainty': float(np.mean(uncertainty)),
            'max_position_uncertainty': float(np.max(uncertainty)),
        }

    def reset(self):
        """重置所有跟踪器"""
        self.trackers = []
        self.frame_count = 0
        SingleObjectKalmanTracker._count = 0

if __name__ == "__main__":
    print("测试多目标卡尔曼跟踪器...")

    tracker = MultiObjectKalmanTracker()

    detections_frame1 = [
        {'x1': 100, 'y1': 100, 'x2': 150, 'y2': 200, 'conf': 0.9, 'class': 0},
        {'x1': 300, 'y1': 150, 'x2': 350, 'y2': 250, 'conf': 0.8, 'class': 0},
    ]

    results = tracker.update(detections_frame1)
    print(f"Frame 1: {len(results)} tracks")

    detections_frame2 = [
        {'x1': 105, 'y1': 105, 'x2': 155, 'y2': 205, 'conf': 0.9, 'class': 0},
        {'x1': 305, 'y1': 155, 'x2': 355, 'y2': 255, 'conf': 0.8, 'class': 0},
    ]
    results = tracker.update(detections_frame2)
    print(f"Frame 2: {len(results)} tracks")

    predictions = tracker.predict()
    print(f"Frame 3 (predict only): {len(predictions)} predictions")
    for pred in predictions:
        print(f"  Track {pred['track_id']}: ({pred['x1']:.1f}, {pred['y1']:.1f}) - ({pred['x2']:.1f}, {pred['y2']:.1f})")

    detections_frame4 = [
        {'x1': 115, 'y1': 115, 'x2': 165, 'y2': 215, 'conf': 0.9, 'class': 0},
        {'x1': 315, 'y1': 165, 'x2': 365, 'y2': 265, 'conf': 0.8, 'class': 0},
    ]
    results = tracker.update(detections_frame4)
    print(f"Frame 4: {len(results)} tracks")

    print("\n✅ 测试完成!")

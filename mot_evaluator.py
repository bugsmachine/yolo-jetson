#!/usr/bin/env python3
"""
MOT (Multiple Object Tracking) 评估模块
计算每帧的 IoU, Precision, Recall 指标


GT 格式 (MOT17):
frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility
"""


import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MOTEvalConfig:
    """评估配置"""
    iou_threshold: float = 0.5  # IoU匹配阈值
    target_class: Optional[int] = 1  # 只评估行人类（class_id=1），None=全部
    min_visibility: float = 0.0  # 最小可见度阈值
    only_not_ignored: bool = True  # 只评估not_ignored=1的目标
    # 坐标变换参数（原始1920x1080 -> 640x640 letterbox）
    original_width: int = 1920
    original_height: int = 1080
    target_size: int = 640

@dataclass
class FrameMetrics:
    """单帧的评估指标"""
    frame_id: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    num_gt: int = 0
    num_det: int = 0
    matched_ious: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.tp + self.fp == 0:
            return 1.0 if self.num_gt == 0 else 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.tp + self.fn == 0:
            return 1.0 if self.num_gt == 0 else 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def mean_iou(self) -> float:
        """平均IoU（只计算匹配的框）"""
        if not self.matched_ious:
            return 0.0
        return np.mean(self.matched_ious)

    @property
    def f1(self) -> float:
        """F1 Score"""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

class MOTEvaluator:
    """
    MOT数据集评估器
    用于计算目标检测的IoU、Precision、Recall
    """

    def __init__(self, gt_file: str, config: MOTEvalConfig = None):
        """
        Args:
            gt_file: gt.txt文件路径
            config: 评估配置
        """
        self.gt_file = gt_file
        self.config = config or MOTEvalConfig()

        self.gt_by_frame: Dict[int, List[Dict]] = defaultdict(list)
        self.frame_metrics: Dict[int, FrameMetrics] = {}

        self._compute_transform_params()

        self._parse_gt_file()

    def _compute_transform_params(self):
        """计算letterbox变换参数"""
        orig_w, orig_h = self.config.original_width, self.config.original_height
        target = self.config.target_size

        scale = min(target / orig_w, target / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        pad_w = (target - new_w) // 2
        pad_h = (target - new_h) // 2

        self.scale = scale
        self.pad_w = pad_w
        self.pad_h = pad_h

        print(f"[MOTEvaluator] 坐标变换参数:")
        print(f"  原始尺寸: {orig_w}x{orig_h}")
        print(f"  目标尺寸: {target}x{target}")
        print(f"  缩放比例: {scale:.4f}")
        print(f"  Padding: ({pad_w}, {pad_h})")

    def _transform_coords(self, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """
        将原始坐标转换为letterbox后的坐标
        输入: x, y, w, h (原始坐标系)
        输出: x1, y1, x2, y2 (转换后的坐标系)
        """

        x_new = x * self.scale + self.pad_w
        y_new = y * self.scale + self.pad_h
        w_new = w * self.scale
        h_new = h * self.scale

        x1 = x_new
        y1 = y_new
        x2 = x_new + w_new
        y2 = y_new + h_new

        return x1, y1, x2, y2

    def _parse_gt_file(self):
        """解析GT文件"""
        print(f"[MOTEvaluator] 解析GT文件: {self.gt_file}")

        total_objects = 0
        filtered_objects = 0

        with open(self.gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                not_ignored = int(parts[6])
                class_id = int(parts[7])
                visibility = float(parts[8])

                total_objects += 1

                if self.config.only_not_ignored and not_ignored == 0:
                    continue
                if self.config.target_class is not None and class_id != self.config.target_class:
                    continue
                if visibility < self.config.min_visibility:
                    continue

                filtered_objects += 1

                x1, y1, x2, y2 = self._transform_coords(x, y, w, h)

                self.gt_by_frame[frame_id].append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'track_id': track_id,
                    'class': class_id,
                    'visibility': visibility,
                })

        print(f"  总对象数: {total_objects}")
        print(f"  过滤后对象数: {filtered_objects}")
        print(f"  帧数: {len(self.gt_by_frame)}")
        print(f"  平均每帧GT: {filtered_objects / len(self.gt_by_frame) if self.gt_by_frame else 0:.2f}")

    @staticmethod
    def compute_iou(box1: Dict, box2: Dict) -> float:
        """计算两个框的IoU"""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def evaluate_frame(self, frame_id: int, detections: List[Dict]) -> FrameMetrics:
        """
        评估单帧

        Args:
            frame_id: 帧ID（从1开始）
            detections: 检测结果列表 [{'x1', 'y1', 'x2', 'y2', 'conf', 'class'}, ...]

        Returns:
            FrameMetrics: 该帧的评估指标
        """
        gt_boxes = self.gt_by_frame.get(frame_id, [])

        metrics = FrameMetrics(frame_id=frame_id)
        metrics.num_gt = len(gt_boxes)
        metrics.num_det = len(detections)

        if not gt_boxes and not detections:
            self.frame_metrics[frame_id] = metrics
            return metrics

        if not gt_boxes:
            metrics.fp = len(detections)
            self.frame_metrics[frame_id] = metrics
            return metrics

        if not detections:
            metrics.fn = len(gt_boxes)
            self.frame_metrics[frame_id] = metrics
            return metrics

        num_gt = len(gt_boxes)
        num_det = len(detections)
        iou_matrix = np.zeros((num_gt, num_det))

        for i, gt in enumerate(gt_boxes):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.compute_iou(gt, det)

        gt_matched = set()
        det_matched = set()
        matched_ious = []

        pairs = []
        for i in range(num_gt):
            for j in range(num_det):
                if iou_matrix[i, j] >= self.config.iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))

        pairs.sort(reverse=True)

        for iou, gt_idx, det_idx in pairs:
            if gt_idx not in gt_matched and det_idx not in det_matched:
                gt_matched.add(gt_idx)
                det_matched.add(det_idx)
                matched_ious.append(iou)

        metrics.tp = len(gt_matched)
        metrics.fp = num_det - len(det_matched)
        metrics.fn = num_gt - len(gt_matched)
        metrics.matched_ious = matched_ious

        self.frame_metrics[frame_id] = metrics
        return metrics

    def get_overall_metrics(self) -> Dict:
        """
        计算所有已评估帧的整体指标

        Returns:
            Dict: 包含precision, recall, f1, mean_iou等指标
        """
        if not self.frame_metrics:
            return {
                'total_frames': 0,
                'total_tp': 0,
                'total_fp': 0,
                'total_fn': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mean_iou': 0.0,
            }

        total_tp = sum(m.tp for m in self.frame_metrics.values())
        total_fp = sum(m.fp for m in self.frame_metrics.values())
        total_fn = sum(m.fn for m in self.frame_metrics.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        all_ious = []
        for m in self.frame_metrics.values():
            all_ious.extend(m.matched_ious)
        mean_iou = np.mean(all_ious) if all_ious else 0.0

        frame_precisions = [m.precision for m in self.frame_metrics.values()]
        frame_recalls = [m.recall for m in self.frame_metrics.values()]
        frame_ious = [m.mean_iou for m in self.frame_metrics.values() if m.matched_ious]

        return {
            'total_frames': len(self.frame_metrics),
            'total_gt': sum(m.num_gt for m in self.frame_metrics.values()),
            'total_det': sum(m.num_det for m in self.frame_metrics.values()),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,

            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': mean_iou,

            'avg_frame_precision': np.mean(frame_precisions) if frame_precisions else 0.0,
            'avg_frame_recall': np.mean(frame_recalls) if frame_recalls else 0.0,
            'avg_frame_iou': np.mean(frame_ious) if frame_ious else 0.0,

            'std_frame_precision': np.std(frame_precisions) if frame_precisions else 0.0,
            'std_frame_recall': np.std(frame_recalls) if frame_recalls else 0.0,
            'std_frame_iou': np.std(frame_ious) if frame_ious else 0.0,
        }

    def print_summary(self):
        """打印评估结果摘要"""
        metrics = self.get_overall_metrics()

        print("\n" + "="*70)
        print("📊 MOT 检测精度评估报告")
        print("="*70)

        print(f"\n【数据统计】")
        print(f"  评估帧数:       {metrics['total_frames']}")
        print(f"  Ground Truth:   {metrics['total_gt']} 个目标")
        print(f"  检测结果:       {metrics['total_det']} 个检测框")

        print(f"\n【匹配结果】 (IoU阈值={self.config.iou_threshold})")
        print(f"  True Positive:  {metrics['total_tp']}")
        print(f"  False Positive: {metrics['total_fp']}")
        print(f"  False Negative: {metrics['total_fn']}")

        print(f"\n【全局指标】")
        print(f"  Precision:      {metrics['precision']*100:.2f}%")
        print(f"  Recall:         {metrics['recall']*100:.2f}%")
        print(f"  F1 Score:       {metrics['f1']*100:.2f}%")
        print(f"  Mean IoU:       {metrics['mean_iou']*100:.2f}%")

        print(f"\n【每帧平均指标】")
        print(f"  Avg Precision:  {metrics['avg_frame_precision']*100:.2f}% (±{metrics['std_frame_precision']*100:.2f}%)")
        print(f"  Avg Recall:     {metrics['avg_frame_recall']*100:.2f}% (±{metrics['std_frame_recall']*100:.2f}%)")
        print(f"  Avg IoU:        {metrics['avg_frame_iou']*100:.2f}% (±{metrics['std_frame_iou']*100:.2f}%)")

        print("="*70)

        return metrics

    def get_per_frame_metrics(self) -> List[Dict]:
        """获取每帧的详细指标（用于绘图）"""
        results = []
        for frame_id in sorted(self.frame_metrics.keys()):
            m = self.frame_metrics[frame_id]
            results.append({
                'frame_id': frame_id,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'mean_iou': m.mean_iou,
                'tp': m.tp,
                'fp': m.fp,
                'fn': m.fn,
                'num_gt': m.num_gt,
                'num_det': m.num_det,
            })
        return results

if __name__ == "__main__":
    import os

    gt_file = "MOT17-10-SDP/gt/gt.txt"

    if os.path.exists(gt_file):
        evaluator = MOTEvaluator(gt_file)

        test_detections = [
            {'x1': 100, 'y1': 200, 'x2': 150, 'y2': 350, 'conf': 0.9, 'class': 0},
            {'x1': 300, 'y1': 250, 'x2': 350, 'y2': 400, 'conf': 0.8, 'class': 0},
        ]

        metrics = evaluator.evaluate_frame(1, test_detections)
        print(f"\n帧1测试结果:")
        print(f"  Precision: {metrics.precision:.2f}")
        print(f"  Recall: {metrics.recall:.2f}")
        print(f"  Mean IoU: {metrics.mean_iou:.2f}")

        evaluator.print_summary()
    else:
        print(f"GT文件不存在: {gt_file}")

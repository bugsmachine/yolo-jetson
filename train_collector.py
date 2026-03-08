#!/usr/bin/env python3
"""
LSTM 训练数据收集模块
用于收集每帧三通道（Kalman/GMC/Inference）的预测结果，
并与 Ground Truth 对比计算精度指标。

输出格式: JSON 文件，包含每帧的 LSTM 分数、运动向量、三通道指标和最佳通道标签。
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import time


@dataclass
class ChannelMetrics:
    """单通道的评估指标"""
    tp: int = 0              # True Positives
    fp: int = 0              # False Positives
    fn: int = 0              # False Negatives
    num_gt: int = 0          # Ground Truth 数量
    num_det: int = 0         # 检测数量
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
    def f1(self) -> float:
        """F1 Score"""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    @property
    def mean_iou(self) -> float:
        """平均 IoU（只计算匹配的框）"""
        if not self.matched_ious:
            return 0.0
        return float(np.mean(self.matched_ious))
    
    def to_dict(self) -> Dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'mean_iou': round(self.mean_iou, 4),
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'num_gt': self.num_gt,
            'num_det': self.num_det,
        }


@dataclass
class ChannelResult:
    """单通道的预测/推理结果"""
    channel: str              # "kalman" | "gmc" | "inference"
    boxes: List[Dict]         # 预测框列表
    latency_ms: float         # 延迟（毫秒）
    source: str = ""          # 详细来源（如 "tier1", "tier2", "tier2_roi"）
    
    def to_dict(self) -> Dict:
        return {
            'channel': self.channel,
            'num_boxes': len(self.boxes),
            'latency_ms': round(self.latency_ms, 2),
            'source': self.source,
            'boxes': self.boxes,
        }


@dataclass
class FrameTrainData:
    """单帧的训练数据"""
    frame_id: int
    lstm_score: float
    motion_vec: Tuple[float, float]
    
    # 三通道结果
    kalman_result: Optional[ChannelResult] = None
    gmc_result: Optional[ChannelResult] = None
    inference_result: Optional[ChannelResult] = None
    
    # 三通道与 GT 对比的指标
    kalman_metrics: Optional[ChannelMetrics] = None
    gmc_metrics: Optional[ChannelMetrics] = None
    inference_metrics: Optional[ChannelMetrics] = None
    
    # Ground Truth 信息
    gt_count: int = 0
    
    # 最佳通道（根据 F1 分数确定）
    best_channel: str = ""
    
    def determine_best_channel(self):
        """根据 F1 分数确定最佳通道"""
        channels = []
        
        if self.kalman_metrics:
            channels.append(('kalman', self.kalman_metrics.f1))
        if self.gmc_metrics:
            channels.append(('gmc', self.gmc_metrics.f1))
        if self.inference_metrics:
            channels.append(('inference', self.inference_metrics.f1))
        
        if channels:
            self.best_channel = max(channels, key=lambda x: x[1])[0]
        else:
            self.best_channel = "unknown"
    
    def to_dict(self) -> Dict:
        """转换为字典（用于 JSON 序列化）"""
        result = {
            'frame_id': self.frame_id,
            'lstm_score': round(self.lstm_score, 4),
            'motion_vec': [round(v, 2) for v in self.motion_vec],
            'gt_count': self.gt_count,
            'best_channel': self.best_channel,
            'channels': {}
        }
        
        if self.kalman_result:
            result['channels']['kalman'] = {
                'result': self.kalman_result.to_dict(),
                'metrics': self.kalman_metrics.to_dict() if self.kalman_metrics else None,
            }
        
        if self.gmc_result:
            result['channels']['gmc'] = {
                'result': self.gmc_result.to_dict(),
                'metrics': self.gmc_metrics.to_dict() if self.gmc_metrics else None,
            }
        
        if self.inference_result:
            result['channels']['inference'] = {
                'result': self.inference_result.to_dict(),
                'metrics': self.inference_metrics.to_dict() if self.inference_metrics else None,
            }
        
        return result


class TrainDataCollector:
    """
    训练数据收集器
    
    职责：
    1. 存储每帧三通道的预测结果
    2. 与 Ground Truth 对比计算指标
    3. 导出训练数据到 JSON 文件
    """
    
    def __init__(self, gt_by_frame: Dict[int, List[Dict]], iou_threshold: float = 0.5):
        """
        Args:
            gt_by_frame: Ground Truth 数据，格式 {frame_id: [{'x1','y1','x2','y2',...}, ...]}
            iou_threshold: IoU 匹配阈值
        """
        self.gt_by_frame = gt_by_frame
        self.iou_threshold = iou_threshold
        self.frame_data: Dict[int, FrameTrainData] = {}
        self.start_time = time.time()
        
        print(f"[TrainCollector] 初始化成功")
        print(f"  - GT 帧数: {len(gt_by_frame)}")
        print(f"  - IoU 阈值: {iou_threshold}")
    
    @staticmethod
    def compute_iou(box1: Dict, box2: Dict) -> float:
        """计算两个框的 IoU"""
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
    
    def evaluate_detections(self, frame_id: int, detections: List[Dict]) -> ChannelMetrics:
        """
        评估检测结果与 GT 的匹配程度
        
        Args:
            frame_id: 帧 ID（MOT 格式，从 1 开始）
            detections: 检测框列表
        
        Returns:
            ChannelMetrics: 评估指标
        """
        gt_boxes = self.gt_by_frame.get(frame_id, [])
        metrics = ChannelMetrics()
        metrics.num_gt = len(gt_boxes)
        metrics.num_det = len(detections)
        
        if not gt_boxes and not detections:
            return metrics
        
        if not gt_boxes:
            metrics.fp = len(detections)
            return metrics
        
        if not detections:
            metrics.fn = len(gt_boxes)
            return metrics
        
        # 计算 IoU 矩阵
        num_gt = len(gt_boxes)
        num_det = len(detections)
        iou_matrix = np.zeros((num_gt, num_det))
        
        for i, gt in enumerate(gt_boxes):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.compute_iou(gt, det)
        
        # 贪婪匹配（按 IoU 降序）
        gt_matched = set()
        det_matched = set()
        matched_ious = []
        
        pairs = []
        for i in range(num_gt):
            for j in range(num_det):
                if iou_matrix[i, j] >= self.iou_threshold:
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
        
        return metrics
    
    def init_frame(self, frame_id: int, lstm_score: float, motion_vec: Tuple[float, float]):
        """初始化帧数据"""
        mot_frame_id = frame_id + 1  # 转换为 MOT 格式（从 1 开始）
        gt_count = len(self.gt_by_frame.get(mot_frame_id, []))
        
        self.frame_data[frame_id] = FrameTrainData(
            frame_id=frame_id,
            lstm_score=lstm_score,
            motion_vec=motion_vec,
            gt_count=gt_count,
        )
    
    def add_kalman_result(self, frame_id: int, boxes: List[Dict], latency_ms: float):
        """添加 Kalman 预测结果"""
        if frame_id not in self.frame_data:
            return
        
        mot_frame_id = frame_id + 1
        
        self.frame_data[frame_id].kalman_result = ChannelResult(
            channel='kalman',
            boxes=boxes,
            latency_ms=latency_ms,
            source='tier0_predict',
        )
        
        self.frame_data[frame_id].kalman_metrics = self.evaluate_detections(mot_frame_id, boxes)
    
    def add_gmc_result(self, frame_id: int, boxes: List[Dict], latency_ms: float):
        """添加 GMC 预测结果"""
        if frame_id not in self.frame_data:
            return
        
        mot_frame_id = frame_id + 1
        
        self.frame_data[frame_id].gmc_result = ChannelResult(
            channel='gmc',
            boxes=boxes,
            latency_ms=latency_ms,
            source='tier0_gmc',
        )
        
        self.frame_data[frame_id].gmc_metrics = self.evaluate_detections(mot_frame_id, boxes)
    
    def add_inference_result(self, frame_id: int, boxes: List[Dict], latency_ms: float, source: str):
        """添加推理结果"""
        if frame_id not in self.frame_data:
            return
        
        mot_frame_id = frame_id + 1
        
        self.frame_data[frame_id].inference_result = ChannelResult(
            channel='inference',
            boxes=boxes,
            latency_ms=latency_ms,
            source=source,
        )
        
        self.frame_data[frame_id].inference_metrics = self.evaluate_detections(mot_frame_id, boxes)
        
        # 确定最佳通道
        self.frame_data[frame_id].determine_best_channel()
    
    def get_summary(self) -> Dict:
        """获取汇总统计"""
        if not self.frame_data:
            return {}
        
        kalman_f1s = []
        gmc_f1s = []
        inference_f1s = []
        best_channel_counts = defaultdict(int)
        
        for frame in self.frame_data.values():
            if frame.kalman_metrics:
                kalman_f1s.append(frame.kalman_metrics.f1)
            if frame.gmc_metrics:
                gmc_f1s.append(frame.gmc_metrics.f1)
            if frame.inference_metrics:
                inference_f1s.append(frame.inference_metrics.f1)
            if frame.best_channel:
                best_channel_counts[frame.best_channel] += 1
        
        total_frames = len(self.frame_data)
        
        return {
            'total_frames': total_frames,
            'elapsed_time_s': round(time.time() - self.start_time, 2),
            
            # 各通道平均 F1
            'avg_kalman_f1': round(np.mean(kalman_f1s), 4) if kalman_f1s else 0.0,
            'avg_gmc_f1': round(np.mean(gmc_f1s), 4) if gmc_f1s else 0.0,
            'avg_inference_f1': round(np.mean(inference_f1s), 4) if inference_f1s else 0.0,
            
            # 各通道 F1 标准差
            'std_kalman_f1': round(np.std(kalman_f1s), 4) if kalman_f1s else 0.0,
            'std_gmc_f1': round(np.std(gmc_f1s), 4) if gmc_f1s else 0.0,
            'std_inference_f1': round(np.std(inference_f1s), 4) if inference_f1s else 0.0,
            
            # 最佳通道分布
            'best_channel_distribution': {
                channel: {
                    'count': count,
                    'ratio': round(count / total_frames, 4)
                }
                for channel, count in best_channel_counts.items()
            },
        }
    
    def save_to_file(self, output_path: str):
        """保存训练数据到 JSON 文件"""
        output_data = {
            'summary': self.get_summary(),
            'config': {
                'iou_threshold': self.iou_threshold,
            },
            'frames': [
                frame.to_dict() 
                for frame in sorted(self.frame_data.values(), key=lambda x: x.frame_id)
            ],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[TrainCollector] 训练数据已保存: {output_path}")
        print(f"  - 总帧数: {len(self.frame_data)}")
        
        summary = self.get_summary()
        print(f"  - 平均 Kalman F1: {summary.get('avg_kalman_f1', 0):.4f}")
        print(f"  - 平均 GMC F1: {summary.get('avg_gmc_f1', 0):.4f}")
        print(f"  - 平均 Inference F1: {summary.get('avg_inference_f1', 0):.4f}")
        
        if 'best_channel_distribution' in summary:
            print(f"  - 最佳通道分布:")
            for channel, stats in summary['best_channel_distribution'].items():
                print(f"      {channel}: {stats['count']} ({stats['ratio']*100:.1f}%)")
    
    def print_progress(self, frame_id: int, interval: int = 50):
        """打印进度（每 interval 帧打印一次）"""
        if frame_id % interval == 0 and frame_id > 0:
            summary = self.get_summary()
            print(f"[TrainCollector] 进度: {len(self.frame_data)} 帧 | "
                  f"F1(K/G/I): {summary.get('avg_kalman_f1', 0):.3f}/"
                  f"{summary.get('avg_gmc_f1', 0):.3f}/"
                  f"{summary.get('avg_inference_f1', 0):.3f}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试 TrainDataCollector...")
    
    # 模拟 GT 数据
    gt_by_frame = {
        1: [{'x1': 100, 'y1': 100, 'x2': 150, 'y2': 200},
            {'x1': 300, 'y1': 150, 'x2': 350, 'y2': 250}],
        2: [{'x1': 105, 'y1': 105, 'x2': 155, 'y2': 205},
            {'x1': 305, 'y1': 155, 'x2': 355, 'y2': 255}],
    }
    
    collector = TrainDataCollector(gt_by_frame, iou_threshold=0.5)
    
    # 模拟帧 0 (对应 MOT frame_id=1)
    collector.init_frame(frame_id=0, lstm_score=0.45, motion_vec=(2.0, 1.0))
    
    # Kalman 预测
    kalman_boxes = [{'x1': 95, 'y1': 95, 'x2': 145, 'y2': 195, 'conf': 0.8, 'class': 0}]
    collector.add_kalman_result(0, kalman_boxes, latency_ms=0.5)
    
    # GMC 预测
    gmc_boxes = [{'x1': 102, 'y1': 102, 'x2': 152, 'y2': 202, 'conf': 0.7, 'class': 0}]
    collector.add_gmc_result(0, gmc_boxes, latency_ms=0.5)
    
    # 推理结果
    inference_boxes = [
        {'x1': 100, 'y1': 100, 'x2': 150, 'y2': 200, 'conf': 0.9, 'class': 0},
        {'x1': 300, 'y1': 150, 'x2': 350, 'y2': 250, 'conf': 0.85, 'class': 0}
    ]
    collector.add_inference_result(0, inference_boxes, latency_ms=15.0, source='tier1')
    
    # 打印结果
    frame = collector.frame_data[0]
    print(f"\n帧 0 结果:")
    print(f"  Best Channel: {frame.best_channel}")
    print(f"  Kalman F1: {frame.kalman_metrics.f1:.3f}")
    print(f"  GMC F1: {frame.gmc_metrics.f1:.3f}")
    print(f"  Inference F1: {frame.inference_metrics.f1:.3f}")
    
    # 保存测试
    collector.save_to_file("test_train_data.json")
    
    print("\n✅ 测试完成!")

#!/usr/bin/env python3
"""
基于 LSTM 时序预测与动态混合精度的端云协同目标检测系统
主程序入口 - 多队列流水线架构

架构设计：
    视频读取线程 → [Frame Queue]
                      ↓
    LSTM Tier0 线程（判断 + 运动估计）
                      ↓
    主决策单元（根据 lstm_score 分发）
        ├→ score < 0.3: 卡尔曼预测（跳过推理）
        ├→ 0.3 < score < 0.7: GMC预测(跳过推理)
        └→ score > 0.7:
            ├→ [YOLO11n Queue] → Tier1 线程
            └→ 如果置信度低 → [YOLO11m Queue] → Tier2 线程
                      ↓
            [Result Queue] → 结果汇总与显示
"""

import cv2
import time
import threading
import queue
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
from collections import defaultdict
import multiprocessing as mp
from multiprocessing import shared_memory, Queue as MPQueue, Event as MPEvent
from multiprocessing.resource_tracker import unregister
import struct

# MOT评估模块
try:
    from mot_evaluator import MOTEvaluator, MOTEvalConfig
    MOT_EVALUATOR_AVAILABLE = True
    print("[Main] ✅ MOT评估模块加载成功")
except ImportError as e:
    MOT_EVALUATOR_AVAILABLE = False
    print(f"[Main] ⚠️ MOT评估模块未找到: {e}")
    print(f"[Main] 请确保 mot_evaluator.py 在同一目录下")
    MOTEvaluator = None
    MOTEvalConfig = None

# 卡尔曼跟踪器模块
try:
    from kalman_tracker import MultiObjectKalmanTracker, KalmanTrackerConfig
    KALMAN_TRACKER_AVAILABLE = True
    print("[Main] ✅ 卡尔曼跟踪器模块加载成功")
except ImportError as e:
    KALMAN_TRACKER_AVAILABLE = False
    print(f"[Main] ⚠️ 卡尔曼跟踪器模块未找到: {e}")
    MultiObjectKalmanTracker = None
    KalmanTrackerConfig = None

# 训练数据收集模块
try:
    from train_collector import TrainDataCollector
    TRAIN_COLLECTOR_AVAILABLE = True
    print("[Main] ✅ 训练数据收集模块加载成功")
except ImportError as e:
    TRAIN_COLLECTOR_AVAILABLE = False
    print(f"[Main] ⚠️ 训练数据收集模块未找到: {e}")
    TrainDataCollector = None


# ==================== 数据结构定义 ====================

class ActionType(Enum):
    """决策动作类型"""
    SKIP_PREDICT = 0    # Case A: 纯卡尔曼预测
    SKIP_GMC = 1        # Case B: 全局运动补偿预测
    INVOKE_TIER1 = 2    # Case C: 激活 YOLO11n
    INVOKE_TIER2 = 3    # Tier1 不确定，激活 YOLO11m
    INVOKE_CLOUD = 4    # Tier2 也不行，发云端

    @property
    def name_str(self):
        """返回字符串名称（用于日志）"""
        names = {
            0: "skip_predict",
            1: "skip_gmc",
            2: "invoke_tier1",
            3: "invoke_tier2",
            4: "invoke_cloud"
        }
        return names.get(self.value, "unknown")


@dataclass
class FramePacket:
    """帧数据包（LSTM输入/输出）"""
    frame_id: int
    image: np.ndarray
    timestamp: float

    # LSTM 输出（初始为空）
    lstm_score: Optional[float] = None      # 画面变化程度 (0-1)
    motion_vec: Optional[Tuple[float, float]] = None  # 全局运动向量 (dx, dy)
    action: Optional[ActionType] = None     # 决策动作

    # Tier1 传递给 Tier2 的 ROI 信息
    roi_boxes: Optional[List[Dict]] = None  # Tier1的检测框（用于ROI裁剪）


@dataclass
class TimingBreakdown:
    """推理时间分解（用于详细性能分析）"""
    preprocess_ms: float = 0.0    # 前处理（resize/normalize/转换格式）
    inference_ms: float = 0.0     # GPU推理（纯模型前向传播）
    postprocess_ms: float = 0.0   # 后处理（NMS/坐标解析/CPU拷贝）
    total_ms: float = 0.0         # 端到端总时间（含队列等待）


@dataclass
class DetectionResult:
    """检测结果（YOLO输出）"""
    frame_id: int
    boxes: List[Dict]  # [{'x1', 'y1', 'x2', 'y2', 'conf', 'class'}, ...]
    source: str        # "tier0_predict" / "tier1" / "tier2" / "tier2_roi" / "cloud"
    latency_ms: float
    timestamp: float
    image: Optional[np.ndarray] = None  # 原始图像（用于显示，可选）
    timing: Optional[TimingBreakdown] = None  # 详细时间分解
    max_conf: float = 0.0  # 最大置信度（用于分析）
    num_boxes: int = 0     # 检测框数量


# ==================== 配置参数 ====================

class Config:
    """系统配置"""
    # ==================== 视频输入 ====================
    VIDEO_PATH = "MOT17-02_640.mp4"
    
    # ==================== MOT评估配置 ====================
    ENABLE_MOT_EVAL = True  # 启用MOT评估
    GT_FILE = "MOT17-02-DPM/gt/gt.txt"  # Ground Truth文件路径
    MOT_IOU_THRESHOLD = 0.5  # IoU匹配阈值
    MOT_TARGET_CLASS = 1  # 目标类别（1=行人），None=全部

    # ==================== 核心目标：能效优化 ====================
    # 处理能力充足（80+ FPS），目标是降低功耗而非提速
    TARGET_FPS = 30              # 目标FPS（与视频帧率一致）
    TARGET_POWER_BUDGET = 7.0    # 目标功耗 7W（vs Baseline 11W）
    TARGET_EARLY_EXIT_RATIO = 0.8  # 目标早退率 80%（跳过推理的帧占比）

    # ==================== 队列大小 ====================
    FRAME_QUEUE_SIZE = 5       # 减小队列
    TIER1_QUEUE_SIZE = 5       # 减小队列
    TIER2_QUEUE_SIZE = 5       # 减小队列
    RESULT_QUEUE_SIZE = 30     # 结果队列

    # ==================== LSTM 阈值（控制早退率） ====================
    LSTM_THRESHOLD_LOW = 0.3   # < 0.3: 纯预测（省电）
    LSTM_THRESHOLD_HIGH = 0.7  # > 0.7: 调用YOLO（保证精度）

    # ==================== YOLO 置信度阈值 ====================
    TIER1_CONF_THRESHOLD = 0.6  # YOLO11n 置信度低于此值 → Tier2
    YOLO_NMS_THRESHOLD = 0.45   # NMS IOU 阈值
    YOLO_CONF_MIN = 0.25        # 最小置信度阈值

    # ==================== 模型路径 ====================
    MODEL_YOLO11N = "yolo11n_int8.engine"  # Tier1模型

    # Tier2双模型策略（全图用TensorRT，ROI用PyTorch）
    MODEL_YOLO11M_FULL = "yolo11m_int8.engine"  # Tier2全图
    MODEL_YOLO11M_ROI = "yolo11m_320_int8.engine"            # Tier2 ROI
    ENABLE_DUAL_MODEL = True                     # 启用双模型策略

    # 模型输入尺寸
    INPUT_SIZE_TIER1 = 640      # Tier1 输入尺寸
    INPUT_SIZE_TIER2_FULL = 640 # Tier2全图输入尺寸
    INPUT_SIZE_TIER2_ROI = 320  # Tier2 ROI输入尺寸

    # ==================== ROI 裁剪配置 ====================
    ENABLE_ROI_CROP = True      # 启用ROI裁剪
    ROI_PADDING_RATIO = 0.2     # ROI边界扩展比例（20%）
    MAX_ROI_COUNT = 3           # 动态策略：ROI超过此数量时改用全图推理
    # 策略：1-3个ROI → ROI推理（快），4+个ROI → 全图推理（避免太多小推理）

    # ==================== 测试模式 ====================
    TEST_MODE_FORCE_TIER1 = False  # 强制所有帧走Tier1（测试用），设为False启用正常调度

    # ==================== 性能模式 ====================
    PERFORMANCE_MODE = False    # 性能测试模式（静默运行，只输出最终结果）

    # ==================== 多进程配置 ====================
    ENABLE_MULTIPROCESSING = True   # 启用多进程架构（绕过GIL）
    SHARED_MEMORY_BUFFER_SIZE = 10  # 共享内存Ring Buffer大小（帧数）
    FRAME_WIDTH = 640               # 帧宽度
    FRAME_HEIGHT = 640              # 帧高度
    FRAME_CHANNELS = 3              # 帧通道数

    # ==================== 功耗估算参数（基于实测） ====================
    # 来源: setup_and_benchmark_models.py 实测数据
    POWER_IDLE = 5.0           # 空闲功耗 (W)
    POWER_YOLO11N = 9.0        # YOLO11n 推理功耗 (W)
    POWER_YOLO11M = 11.0       # YOLO11m 推理功耗 (W)
    POWER_LSTM = 5.5           # LSTM + 卡尔曼预测功耗 (W)

    # ==================== 性能统计与调试 ====================
    ENABLE_STATS = True         # 启用性能统计
    STATS_PRINT_INTERVAL = 30   # 每N帧打印一次统计
    DISPLAY_OUTPUT = False      # 是否显示画面（调试用）
    SAVE_LOGS = True           # 保存日志到文件

    # ==================== 训练模式配置 ====================
    TRAIN_MODE = True                           # LSTM训练数据收集模式
    TRAIN_DATA_OUTPUT = "train_data.json"       # 训练数据输出文件
    TRAIN_PROGRESS_INTERVAL = 50                # 训练进度打印间隔

    # ==================== 策略配置 ====================
    WARMUP_FRAMES = 5           # 启动前N帧强制走推理（等待Tracker稳定）
    
    # ==================== 运行时视频参数 ====================
    VIDEO_WIDTH = 1920          # 默认 1920 (运行时可能会更新)
    VIDEO_HEIGHT = 1080         # 默认 1080


# ==================== 共享内存管理器 ====================

class SharedMemoryFrameBuffer:
    """
    共享内存Ring Buffer（零拷贝帧传输）
    设计思路：
    - 共享内存存储多个帧槽（Ring Buffer）
    - 使用Queue传递索引号（轻量级通知）
    - 生产者(VideoReader)写帧，消费者(main)读帧
    """
    def __init__(self, buffer_size: int, width: int, height: int, channels: int, create: bool = True):
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.channels = channels
        self.frame_size = width * height * channels  # 每帧字节数
        self.total_size = self.frame_size * buffer_size

        # 每个槽的元数据大小（frame_id: int64, timestamp: float64, lstm_score: float64, action: int32）
        self.metadata_size = 8 + 8 + 8 + 4  # 28字节
        self.slot_size = self.frame_size + self.metadata_size
        self.total_size = self.slot_size * buffer_size

        if create:
            # 创建共享内存
            self.shm = shared_memory.SharedMemory(
                name="frame_buffer_shm",
                create=True,
                size=self.total_size
            )
            print(f"[SharedMemory] 创建成功: {self.total_size / 1024 / 1024:.2f} MB")
        else:
            # 连接到已存在的共享内存
            self.shm = shared_memory.SharedMemory(name="frame_buffer_shm")
            unregister(self.shm._name, "shared_memory")
            print(f"[SharedMemory] 连接成功")

        # 使用numpy视图访问共享内存
        self.buffer = np.ndarray(
            (buffer_size, height, width, channels),
            dtype=np.uint8,
            buffer=self.shm.buf[:self.frame_size * buffer_size]
        )

        # 元数据缓冲区
        self.metadata_buffer = self.shm.buf[self.frame_size * buffer_size:]

    def write_frame(self, slot_idx: int, frame: np.ndarray, frame_id: int,
                    timestamp: float, lstm_score: float = 0.0, action: int = 0):
        """写入帧到指定槽位"""
        # 验证形状
        if frame.shape != (self.height, self.width, self.channels):
            # 如果尺寸不匹配，简单resize (防止崩溃)
            # print(f"[SharedMemory] ⚠️ 帧尺寸不匹配 {frame.shape} vs {(self.height, self.width, self.channels)}，正在 resize")
            frame = cv2.resize(frame, (self.width, self.height))
        
        # 写入图像数据
        self.buffer[slot_idx] = frame

        # 写入元数据
        offset = slot_idx * self.metadata_size
        struct.pack_into('qddi', self.metadata_buffer, offset,
                        frame_id, timestamp, lstm_score, action)

    def read_frame(self, slot_idx: int) -> Tuple[np.ndarray, int, float, float, int]:
        """从指定槽位读取帧"""
        # 读取图像数据（拷贝一份，避免被覆盖）
        frame = self.buffer[slot_idx].copy()

        # 读取元数据
        offset = slot_idx * self.metadata_size
        frame_id, timestamp, lstm_score, action = struct.unpack_from(
            'qddi', self.metadata_buffer, offset
        )

        return frame, frame_id, timestamp, lstm_score, action

    def close(self):
        """关闭共享内存"""
        try:
            # 先释放numpy数组引用
            if hasattr(self, 'buffer'):
                del self.buffer
            # 释放memoryview引用
            if hasattr(self, 'metadata_buffer'):
                del self.metadata_buffer
            # 关闭共享内存
            if hasattr(self, 'shm') and self.shm is not None:
                self.shm.close()
                self.shm = None
        except Exception as e:
            print(f"[SharedMemory] close警告: {e}")

    def unlink(self):
        """删除共享内存"""
        try:
            if hasattr(self, 'shm') and self.shm is not None:
                self.shm.unlink()
        except FileNotFoundError:
            pass  # 已经被删除了
        except Exception as e:
            print(f"[SharedMemory] unlink警告: {e}")

    def __del__(self):
        """析构函数：确保资源释放"""
        try:
            self.close()
        except Exception:
            pass


# ==================== 全局队列 ====================

frame_queue = queue.Queue(maxsize=Config.FRAME_QUEUE_SIZE)      # 原始帧队列
tier1_queue = queue.Queue(maxsize=Config.TIER1_QUEUE_SIZE)      # YOLO11n 队列
tier2_queue = queue.Queue(maxsize=Config.TIER2_QUEUE_SIZE)      # YOLO11m 队列
result_queue = queue.Queue(maxsize=Config.RESULT_QUEUE_SIZE)    # 结果队列

# 全局状态
stop_event = threading.Event()  # 停止信号

# 🆕 共享检测结果状态（用于Tier0运动预测）
import threading
latest_detection_lock = threading.Lock()
latest_detection_boxes: List[Dict] = []  # 最新的检测框列表
latest_detection_frame_id: int = -1       # 对应的帧ID

# 🆕 全局卡尔曼跟踪器（用于Tier0卡尔曼预测）
kalman_tracker_lock = threading.Lock()
global_kalman_tracker = None  # 延迟初始化
if KALMAN_TRACKER_AVAILABLE:
    global_kalman_tracker = MultiObjectKalmanTracker(KalmanTrackerConfig(
        max_age=30,           # 最大丢失帧数
        min_hits=2,           # 最小命中次数（降低，更快确认）
        iou_threshold=0.3,    # IoU匹配阈值
    ))
    print("[Main] ✅ 全局卡尔曼跟踪器初始化成功")

# 🆕 训练模式相关全局变量
train_result_queue = queue.Queue(maxsize=100)  # 训练模式：推理结果回传队列
train_collector_lock = threading.Lock()
global_train_collector = None  # 延迟初始化（需要 GT 数据）



# ==================== Tier 0: LSTM + 运动估计线程 ====================

class Tier0_LSTM_Thread(threading.Thread):
    """
    Tier 0: LSTM 语义引导 + 运动估计
    职责：
    1. 从 frame_queue 取帧
    2. LSTM 判断画面变化程度 → lstm_score
    3. 光流法计算全局运动向量 → motion_vec
    4. 决策 action 并传递给主决策单元
    """
    def __init__(self):
        super().__init__(daemon=True, name="Tier0-LSTM")
        self.prev_frame = None
        
        # 🆕 光流相关状态
        self.prev_gray = None           # 上一帧灰度图（用于光流）
        self.prev_points = None         # 上一帧特征点
        self.lk_params = dict(
            winSize=(21, 21),            # 搜索窗口大小
            maxLevel=3,                  # 金字塔层数
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.feature_params = dict(
            maxCorners=100,              # 最多检测100个角点
            qualityLevel=0.3,            # 角点质量阈值
            minDistance=7,               # 角点最小间距
            blockSize=7                  # 检测块大小
        )

    def _lstm_predict_score(self, frame: np.ndarray) -> float:
        """
        【Placeholder】LSTM 预测画面变化程度
        TODO: 实现轻量级 LSTM 网络
        """
        # 临时实现：简单帧差
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 1.0  # 第一帧强制推理

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.prev_frame, gray)
        score = np.mean(diff) / 255.0

        self.prev_frame = gray
        return float(score)

    def _estimate_motion_vector(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        估计全局运动向量（使用稀疏光流 Lucas-Kanade）
        返回: (dx, dy) 全局运动向量（像素）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 第一帧：初始化特征点
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            return (0.0, 0.0)
        
        # 如果没有追踪点，重新检测
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            if self.prev_points is None:
                self.prev_gray = gray
                return (0.0, 0.0)
        
        # 使用光流追踪特征点
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if next_points is None:
            self.prev_gray = gray
            self.prev_points = None
            return (0.0, 0.0)
        
        # 筛选成功追踪的点
        status = status.flatten()
        good_old = self.prev_points[status == 1].reshape(-1, 2)
        good_new = next_points[status == 1].reshape(-1, 2)
        
        if len(good_old) < 5:
            # 追踪点太少，重新检测
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            return (0.0, 0.0)
        
        # 计算运动向量（使用中位数以抵消异常值）
        motion_vectors = good_new - good_old
        dx = float(np.median(motion_vectors[:, 0]))
        dy = float(np.median(motion_vectors[:, 1]))
        
        # 更新状态
        self.prev_gray = gray
        # 定期刷新特征点（每10帧或追踪点减少时）
        if len(good_new) < 30:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
        else:
            self.prev_points = good_new.reshape(-1, 1, 2)
        
        return (dx, dy)

    def _decide_action(self, lstm_score: float, frame_id: int) -> ActionType:
        """根据 lstm_score 决定动作"""
        # Warmup期：强制推理（初始化Tracker）
        if frame_id < Config.WARMUP_FRAMES:
            return ActionType.INVOKE_TIER1

        # 测试模式：强制所有帧走Tier1
        if Config.TEST_MODE_FORCE_TIER1:
            return ActionType.INVOKE_TIER1

        # 正常模式：根据LSTM分数决策
        if lstm_score < Config.LSTM_THRESHOLD_LOW:
            return ActionType.SKIP_PREDICT
        elif lstm_score < Config.LSTM_THRESHOLD_HIGH:
            return ActionType.SKIP_GMC
        else:
            return ActionType.INVOKE_TIER1

    def run(self):
        print("[Tier0-LSTM] 线程启动")
        
        if Config.TRAIN_MODE:
            print("[Tier0-LSTM] ⭐ 训练模式启用：每帧将执行三通道（Kalman/GMC/Inference）")
        
        while not stop_event.is_set():
            try:
                # 从帧队列取数据（超时1秒）
                packet: FramePacket = frame_queue.get(timeout=1.0)

                # 1. LSTM 预测
                t0 = time.perf_counter()
                lstm_score = self._lstm_predict_score(packet.image)

                # 2. 运动估计
                motion_vec = self._estimate_motion_vector(packet.image)

                latency = (time.perf_counter() - t0) * 1000

                # 更新 packet
                packet.lstm_score = lstm_score
                packet.motion_vec = motion_vec

                # ========== 训练模式：三通道并行执行 ==========
                if Config.TRAIN_MODE:
                    self._run_train_mode(packet, lstm_score, motion_vec)
                else:
                    # ========== 正常模式 ==========
                    # 3. 决策
                    action = self._decide_action(lstm_score, packet.frame_id)
                    packet.action = action

                    # 4. 根据决策分发
                    if action == ActionType.SKIP_PREDICT or action == ActionType.SKIP_GMC:
                        # 不需要推理，直接生成预测结果
                        result = self._generate_prediction_result(packet)
                        result_queue.put(result)

                    elif action == ActionType.INVOKE_TIER1:
                        # 需要推理，放入 Tier1 队列
                        tier1_queue.put(packet)

                    if Config.ENABLE_STATS and packet.frame_id % Config.STATS_PRINT_INTERVAL == 0:
                        print(f"[Tier0] Frame {packet.frame_id}: score={lstm_score:.3f}, "
                              f"action={action.name_str}, latency={latency:.2f}ms, "
                              f"queue(t1/t2): {tier1_queue.qsize()}/{tier2_queue.qsize()}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Tier0-LSTM] 错误: {e}")
                import traceback
                traceback.print_exc()

        print("[Tier0-LSTM] 线程结束")
    
    def _run_train_mode(self, packet: FramePacket, lstm_score: float, motion_vec: Tuple[float, float]):
        """
        训练模式：同时执行三个通道并收集数据
        
        1. Kalman 预测
        2. GMC 预测  
        3. Tier1/Tier2 推理（阻塞等待结果）
        """
        global global_train_collector, train_collector_lock
        
        frame_id = packet.frame_id
        
        # 初始化帧数据
        if global_train_collector is not None:
            with train_collector_lock:
                global_train_collector.init_frame(frame_id, lstm_score, motion_vec)
        
        # ========== 1. Kalman 预测 ==========
        t_kalman_start = time.perf_counter()
        
        # 临时设置 action 以使用现有方法
        packet.action = ActionType.SKIP_PREDICT
        kalman_result = self._generate_kalman_prediction(packet)
        
        kalman_latency = (time.perf_counter() - t_kalman_start) * 1000
        
        if global_train_collector is not None:
            with train_collector_lock:
                global_train_collector.add_kalman_result(
                    frame_id, 
                    kalman_result.boxes, 
                    kalman_latency
                )
        
        # ========== 2. GMC 预测 ==========
        t_gmc_start = time.perf_counter()
        
        packet.action = ActionType.SKIP_GMC
        gmc_result = self._generate_gmc_prediction(packet)
        
        gmc_latency = (time.perf_counter() - t_gmc_start) * 1000
        
        if global_train_collector is not None:
            with train_collector_lock:
                global_train_collector.add_gmc_result(
                    frame_id,
                    gmc_result.boxes,
                    gmc_latency
                )
        
        # ========== 3. Tier1/Tier2 推理 ==========
        # 标记为训练模式帧，让推理线程知道需要回传结果
        packet.action = ActionType.INVOKE_TIER1
        
        # 将帧 ID 存入特殊属性，用于结果回收
        if not hasattr(packet, 'train_mode_flag'):
            # 使用动态属性标记
            pass
        
        # 发送到推理队列
        tier1_queue.put(packet)
        
        # 等待推理结果（从 train_result_queue 获取）
        # 注意：推理线程会同时将结果放入 result_queue 和 train_result_queue
        try:
            inference_result = train_result_queue.get(timeout=10.0)
            
            # 确保是当前帧的结果
            while inference_result.frame_id != frame_id:
                # 结果顺序可能乱序，暂时跳过（简化处理）
                inference_result = train_result_queue.get(timeout=5.0)
            
            if global_train_collector is not None:
                with train_collector_lock:
                    global_train_collector.add_inference_result(
                        frame_id,
                        inference_result.boxes,
                        inference_result.latency_ms,
                        inference_result.source
                    )
                    
                    # 打印进度
                    global_train_collector.print_progress(
                        frame_id, 
                        Config.TRAIN_PROGRESS_INTERVAL
                    )
                    
        except queue.Empty:
            print(f"[Tier0-Train] ⚠️ Frame {frame_id}: 等待推理结果超时")


    def _generate_prediction_result(self, packet: FramePacket) -> DetectionResult:
        """
        根据action类型生成预测结果：
        - SKIP_PREDICT: 使用卡尔曼滤波预测
        - SKIP_GMC: 使用全局运动补偿(GMC)预测
        """
        if packet.action == ActionType.SKIP_PREDICT:
            return self._generate_kalman_prediction(packet)
        else:  # SKIP_GMC
            return self._generate_gmc_prediction(packet)
    
    def _generate_kalman_prediction(self, packet: FramePacket) -> DetectionResult:
        """
        使用卡尔曼滤波器预测目标位置
        基于目标的历史轨迹和速度模型预测下一帧位置
        """
        global global_kalman_tracker, kalman_tracker_lock
        
        predicted_boxes = []
        
        if global_kalman_tracker is not None:
            with kalman_tracker_lock:
                # 卡尔曼预测（不更新，纯预测）
                predicted_boxes = global_kalman_tracker.predict()
        
        # 获取图像尺寸（用于边界限制）
        h, w = packet.image.shape[:2]
        
        # 边界裁剪
        clipped_boxes = []
        for box in predicted_boxes:
            x1 = max(0, min(box['x1'], w - 1))
            y1 = max(0, min(box['y1'], h - 1))
            x2 = max(0, min(box['x2'], w - 1))
            y2 = max(0, min(box['y2'], h - 1))
            
            if x2 > x1 and y2 > y1:
                clipped_boxes.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'conf': box.get('conf', 0.5),
                    'class': box.get('class', 0),
                    'track_id': box.get('track_id', -1)
                })
        
        if Config.ENABLE_STATS and not Config.PERFORMANCE_MODE:
            tracker_count = global_kalman_tracker.get_tracker_count() if global_kalman_tracker else 0
            print(f"[Tier0-Kalman] Frame {packet.frame_id}: "
                  f"predicted {len(clipped_boxes)} boxes, {tracker_count} trackers active")
        
        return DetectionResult(
            frame_id=packet.frame_id,
            boxes=clipped_boxes,
            source="tier0_predict",
            latency_ms=0.5,
            timestamp=packet.timestamp,
            max_conf=max([b['conf'] for b in clipped_boxes]) if clipped_boxes else 0.0,
            num_boxes=len(clipped_boxes)
        )
    
    def _generate_gmc_prediction(self, packet: FramePacket) -> DetectionResult:
        """
        使用全局运动补偿(GMC)预测目标位置
        基于光流计算的全局运动向量平移检测框
        """
        # 获取最新的检测框
        global latest_detection_boxes, latest_detection_frame_id, latest_detection_lock
        
        with latest_detection_lock:
            prev_boxes = latest_detection_boxes.copy()  # 深拷贝避免竞争
            prev_frame_id = latest_detection_frame_id
        
        # 如果没有历史检测框，返回空结果
        if not prev_boxes or prev_frame_id < 0:
            return DetectionResult(
                frame_id=packet.frame_id,
                boxes=[],
                source="tier0_gmc",
                latency_ms=0.5,
                timestamp=packet.timestamp
            )
        
        # 使用运动向量平移检测框
        motion_vec = packet.motion_vec if packet.motion_vec else (0.0, 0.0)
        dx, dy = motion_vec
        
        # 获取图像尺寸（用于边界限制）
        h, w = packet.image.shape[:2]
        
        predicted_boxes = []
        for box in prev_boxes:
            # 平移框坐标
            new_x1 = box['x1'] + dx
            new_y1 = box['y1'] + dy
            new_x2 = box['x2'] + dx
            new_y2 = box['y2'] + dy
            
            # 边界检查：确保框不会完全移出画面
            if new_x2 > 0 and new_x1 < w and new_y2 > 0 and new_y1 < h:
                # 裁剪到图像边界内
                new_x1 = max(0, min(new_x1, w - 1))
                new_y1 = max(0, min(new_y1, h - 1))
                new_x2 = max(0, min(new_x2, w - 1))
                new_y2 = max(0, min(new_y2, h - 1))
                
                # 确保框的宽高有效
                if new_x2 > new_x1 and new_y2 > new_y1:
                    predicted_boxes.append({
                        'x1': float(new_x1),
                        'y1': float(new_y1),
                        'x2': float(new_x2),
                        'y2': float(new_y2),
                        'conf': box['conf'] * 0.9,  # 预测框置信度衰减
                        'class': box['class']
                    })
        
        if Config.ENABLE_STATS and not Config.PERFORMANCE_MODE:
            print(f"[Tier0-GMC] Frame {packet.frame_id}: motion=({dx:.1f}, {dy:.1f}), "
                  f"predicted {len(predicted_boxes)} boxes from frame {prev_frame_id}")
        
        return DetectionResult(
            frame_id=packet.frame_id,
            boxes=predicted_boxes,
            source="tier0_gmc",
            latency_ms=0.5,
            timestamp=packet.timestamp,
            max_conf=max([b['conf'] for b in predicted_boxes]) if predicted_boxes else 0.0,
            num_boxes=len(predicted_boxes)
        )


# ==================== Tier 1: YOLO11n 线程 ====================

class Tier1_YOLO_Thread(threading.Thread):
    """
    Tier 1: YOLO11n 快速全局检测
    职责：
    1. 从 tier1_queue 取帧
    2. 运行 YOLO11n 推理
    3. 高置信度 → 直接输出
    4. 低置信度 → 放入 tier2_queue
    """
    def __init__(self):
        super().__init__(daemon=True, name="Tier1-YOLO11n")
        self.model = None
        self.ready_event = threading.Event()  # 模型就绪标志

    def _load_model(self):
        """加载 YOLO11n 模型（支持TensorRT/PyTorch）"""
        print(f"[Tier1] 加载 YOLO11n 模型: {Config.MODEL_YOLO11N}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(Config.MODEL_YOLO11N)
            print(f"[Tier1] 模型加载完成 ({'TensorRT' if Config.MODEL_YOLO11N.endswith('.engine') else 'PyTorch'})")

            # Warmup：预热模型，避免首次推理的GPU初始化延迟
            print("[Tier1] 正在Warmup模型...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for i in range(3):
                self.model(dummy_img, imgsz=Config.INPUT_SIZE_TIER1, verbose=False)
            print("[Tier1] Warmup完成")

        except Exception as e:
            print(f"[Tier1] ⚠️  模型加载失败: {e}")
            print(f"[Tier1] 使用模拟推理模式")
            self.model = None
        finally:
            # 无论成功或失败，都标记为就绪
            self.ready_event.set()
            print("[Tier1] 模型就绪，可以开始推理")

    def _inference(self, image: np.ndarray) -> Tuple[List[Dict], float, TimingBreakdown]:
        """
        YOLO11n 推理
        返回: (boxes, max_confidence, timing_breakdown)
        """
        timing = TimingBreakdown()

        if self.model is None:
            # 模拟推理（测试用）
            time.sleep(0.010)
            boxes = []
            max_conf = np.random.uniform(0.3, 0.9)
            timing.total_ms = 10.0
            timing.inference_ms = 8.0
            timing.preprocess_ms = 1.0
            timing.postprocess_ms = 1.0
            return boxes, max_conf, timing

        # 真实推理
        t_total_start = time.perf_counter()
        results = self.model(
            image,
            imgsz=Config.INPUT_SIZE_TIER1,
            conf=Config.YOLO_CONF_MIN,
            iou=Config.YOLO_NMS_THRESHOLD,
            verbose=False
        )

        # 解析结果
        t_parse_start = time.perf_counter()
        boxes = self._parse_results(results[0])
        t_parse_end = time.perf_counter()

        max_conf = max([b['conf'] for b in boxes]) if boxes else 0.0

        # 使用YOLO内置的speed属性获取时间分解
        # speed字典包含: {'preprocess': x, 'inference': x, 'postprocess': x}
        speed = results[0].speed
        timing.preprocess_ms = speed.get('preprocess', 0.0)
        timing.inference_ms = speed.get('inference', 0.0)
        # 后处理 = YOLO内部postprocess + 我们的解析时间
        timing.postprocess_ms = speed.get('postprocess', 0.0) + (t_parse_end - t_parse_start) * 1000
        timing.total_ms = (t_parse_end - t_total_start) * 1000

        return boxes, max_conf, timing

    def _parse_results(self, result) -> List[Dict]:
        """解析YOLO结果为统一格式"""
        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # 提取坐标 (xyxy格式)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                boxes.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'conf': conf,
                    'class': cls
                })

        return boxes

    def run(self):
        self._load_model()
        if not Config.PERFORMANCE_MODE:
            print("[Tier1-YOLO11n] 线程启动")

        while not stop_event.is_set():
            try:
                packet: FramePacket = tier1_queue.get(timeout=1.0)

                # 推理（返回详细时间分解）
                boxes, max_conf, timing = self._inference(packet.image)

                # 判断是否需要 Tier2
                if max_conf < Config.TIER1_CONF_THRESHOLD:
                    # 置信度不够，放入 Tier2 队列
                    # 动态ROI策略：根据检测框数量决定是否传ROI
                    roi_count = len(boxes)

                    if roi_count > 0 and roi_count <= Config.MAX_ROI_COUNT:
                        # ROI数量合理（1-3个），传递ROI给Tier2
                        packet.roi_boxes = boxes
                        strategy = f"ROI推理({roi_count}个)"
                    else:
                        # ROI数量过多（>3个）或没有框，清空ROI改用全图推理
                        packet.roi_boxes = None
                        strategy = f"全图推理(ROI过多:{roi_count})" if roi_count > Config.MAX_ROI_COUNT else "全图推理(无框)"

                    if Config.ENABLE_STATS and not Config.PERFORMANCE_MODE:
                        print(f"[Tier1] Frame {packet.frame_id}: max_conf={max_conf:.2f} < {Config.TIER1_CONF_THRESHOLD} → Tier2 [{strategy}]")
                    tier2_queue.put(packet)
                else:
                    # 置信度足够，直接输出
                    result = DetectionResult(
                        frame_id=packet.frame_id,
                        boxes=boxes,
                        source="tier1",
                        latency_ms=timing.total_ms,
                        timestamp=packet.timestamp,
                        timing=timing,
                        max_conf=max_conf,
                        num_boxes=len(boxes)
                    )
                    result_queue.put(result)
                    
                    # 🆕 训练模式：同时放入训练结果队列
                    if Config.TRAIN_MODE:
                        train_result_queue.put(result)

                    if Config.ENABLE_STATS and not Config.PERFORMANCE_MODE:
                        print(f"[Tier1] Frame {packet.frame_id}: max_conf={max_conf:.2f}, "
                              f"total={timing.total_ms:.2f}ms (pre={timing.preprocess_ms:.1f}, "
                              f"inf={timing.inference_ms:.1f}, post={timing.postprocess_ms:.1f})")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Tier1-YOLO11n] 错误: {e}")
                import traceback
                traceback.print_exc()

        print("[Tier1-YOLO11n] 线程结束")


# ==================== Tier 2: YOLO11m 线程 ====================

class Tier2_YOLO_Thread(threading.Thread):
    """
    Tier 2: YOLO11m-INT8 精确检测
    职责：
    1. 从 tier2_queue 取帧
    2. 运行 YOLO11m 推理（可选 ROI 裁剪）
    3. 输出最终结果
    """
    def __init__(self):
        super().__init__(daemon=True, name="Tier2-YOLO11m")
        self.model_full = None   # 全图推理模型
        self.model_roi = None    # ROI推理模型
        self.ready_event = threading.Event()  # 模型就绪标志

        # 性能统计
        self.stats_full = []     # 全图推理延迟统计
        self.stats_roi = []      # ROI推理延迟统计

    def _load_model(self):
        """加载 YOLO11m 双模型（全图TensorRT + ROI）"""
        from ultralytics import YOLO

        if not Config.PERFORMANCE_MODE:
            print(f"[Tier2] 🔥 双模型策略启用" if Config.ENABLE_DUAL_MODEL else f"[Tier2] 加载单模型")

        try:
            # 1️⃣ 加载全图模型（TensorRT）
            if not Config.PERFORMANCE_MODE:
                print(f"[Tier2-Full] 加载全图模型: {Config.MODEL_YOLO11M_FULL}")
            self.model_full = YOLO(Config.MODEL_YOLO11M_FULL)
            if not Config.PERFORMANCE_MODE:
                print(f"[Tier2-Full] 模型加载完成 (TensorRT)")

            # Warmup全图模型
            if not Config.PERFORMANCE_MODE:
                print("[Tier2-Full] 正在Warmup...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(3):
                self.model_full(dummy_img, imgsz=Config.INPUT_SIZE_TIER2_FULL, verbose=False)
            if not Config.PERFORMANCE_MODE:
                print("[Tier2-Full] Warmup完成")

            # 2️⃣ 加载ROI模型（PyTorch，可选）
            if Config.ENABLE_DUAL_MODEL:
                if not Config.PERFORMANCE_MODE:
                    print(f"[Tier2-ROI] 加载ROI模型: {Config.MODEL_YOLO11M_ROI}")
                self.model_roi = YOLO(Config.MODEL_YOLO11M_ROI)
                if not Config.PERFORMANCE_MODE:
                    print(f"[Tier2-ROI] 模型加载完成")

                # Warmup ROI模型（PyTorch需要更多Warmup）
                if not Config.PERFORMANCE_MODE:
                    print("[Tier2-ROI] 正在Warmup...")
                dummy_roi = np.zeros((320, 320, 3), dtype=np.uint8)
                for i in range(10):  
                    self.model_roi(dummy_roi, imgsz=Config.INPUT_SIZE_TIER2_ROI, verbose=False)
                    if not Config.PERFORMANCE_MODE and i % 3 == 0:
                        print(f"[Tier2-ROI] Warmup进度: {i+1}/10")
                if not Config.PERFORMANCE_MODE:
                    print("[Tier2-ROI] Warmup完成")

        except Exception as e:
            if not Config.PERFORMANCE_MODE:
                print(f"[Tier2] ⚠️  模型加载失败: {e}")
                print(f"[Tier2] 使用模拟推理模式")
            self.model_full = None
            self.model_roi = None
        finally:
            # 无论成功或失败，都标记为就绪
            self.ready_event.set()
            if not Config.PERFORMANCE_MODE:
                print("[Tier2] 模型就绪，可以开始推理")

    def _inference(self, image: np.ndarray, roi_boxes: Optional[List[Dict]] = None) -> Tuple[List[Dict], str, TimingBreakdown]:
        """
        YOLO11m 双模型推理（全图TensorRT + ROI PyTorch）
        Args:
            image: 原始图像
            roi_boxes: Tier1提供的ROI区域（可选）
        Returns:
            (检测框列表, 使用的模型类型, 时间分解)
        """
        timing = TimingBreakdown()

        if self.model_full is None:
            # 模拟推理（测试用）
            time.sleep(0.020)
            timing.total_ms = 20.0
            timing.inference_ms = 16.0
            timing.preprocess_ms = 2.0
            timing.postprocess_ms = 2.0
            return [], "sim", timing

        # 双模型策略选择
        if Config.ENABLE_DUAL_MODEL and Config.ENABLE_ROI_CROP and roi_boxes and len(roi_boxes) > 0 and self.model_roi is not None:
            # 使用ROI模型（小尺寸，快）
            boxes, timing = self._inference_with_roi(image, roi_boxes)
            return boxes, "roi", timing
        else:
            # 使用TensorRT全图模型（标准）
            boxes, timing = self._inference_full(image)
            return boxes, "full", timing

    def _inference_full(self, image: np.ndarray) -> Tuple[List[Dict], TimingBreakdown]:
        """全图推理（TensorRT引擎）"""
        timing = TimingBreakdown()
        t_total_start = time.perf_counter()

        results = self.model_full(
            image,
            imgsz=Config.INPUT_SIZE_TIER2_FULL,
            conf=Config.YOLO_CONF_MIN,
            iou=Config.YOLO_NMS_THRESHOLD,
            verbose=False
        )

        t_parse_start = time.perf_counter()
        boxes = self._parse_results(results[0])
        t_parse_end = time.perf_counter()

        # 使用YOLO内置speed属性
        speed = results[0].speed
        timing.preprocess_ms = speed.get('preprocess', 0.0)
        timing.inference_ms = speed.get('inference', 0.0)
        timing.postprocess_ms = speed.get('postprocess', 0.0) + (t_parse_end - t_parse_start) * 1000
        timing.total_ms = (t_parse_end - t_total_start) * 1000

        self.stats_full.append(timing.total_ms)  # 记录统计
        return boxes, timing

    def _inference_with_roi(self, image: np.ndarray, roi_boxes: List[Dict]) -> Tuple[List[Dict], TimingBreakdown]:
        """
        ROI裁剪推理（PyTorch模型，支持小尺寸320×320）
        将每个ROI区域裁剪后单独推理，大幅降低计算量
        """
        timing = TimingBreakdown()
        t_total_start = time.perf_counter()

        h, w = image.shape[:2]
        all_boxes = []
        total_preprocess = 0.0
        total_inference = 0.0
        total_postprocess = 0.0

        for roi in roi_boxes:
            # 1. 扩展ROI边界（避免边缘目标被裁掉）
            x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
            roi_w, roi_h = x2 - x1, y2 - y1

            # 扩展padding
            pad_w = roi_w * Config.ROI_PADDING_RATIO
            pad_h = roi_h * Config.ROI_PADDING_RATIO

            # 裁剪坐标（确保不越界）
            crop_x1 = max(0, int(x1 - pad_w))
            crop_y1 = max(0, int(y1 - pad_h))
            crop_x2 = min(w, int(x2 + pad_w))
            crop_y2 = min(h, int(y2 + pad_h))

            # 2. 裁剪ROI图像
            roi_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # 3. 使用ROI模型推理（小尺寸320×320）
            results = self.model_roi(
                roi_image,
                imgsz=Config.INPUT_SIZE_TIER2_ROI,
                conf=Config.YOLO_CONF_MIN,
                iou=Config.YOLO_NMS_THRESHOLD,
                verbose=False
            )

            # 累加每个ROI的时间
            speed = results[0].speed
            total_preprocess += speed.get('preprocess', 0.0)
            total_inference += speed.get('inference', 0.0)
            total_postprocess += speed.get('postprocess', 0.0)

            # 4. 解析结果并映射回全图坐标
            roi_detections = self._parse_results(results[0])
            for box in roi_detections:
                box['x1'] += crop_x1
                box['y1'] += crop_y1
                box['x2'] += crop_x1
                box['y2'] += crop_y1
                all_boxes.append(box)

        # 5. 全局NMS（去除跨ROI的重复检测）
        t_nms_start = time.perf_counter()
        all_boxes = self._nms_boxes(all_boxes)
        t_nms_end = time.perf_counter()

        timing.preprocess_ms = total_preprocess
        timing.inference_ms = total_inference
        timing.postprocess_ms = total_postprocess + (t_nms_end - t_nms_start) * 1000
        timing.total_ms = (t_nms_end - t_total_start) * 1000

        self.stats_roi.append(timing.total_ms)  # 记录统计

        return all_boxes, timing

    def _parse_results(self, result) -> List[Dict]:
        """解析YOLO结果为统一格式"""
        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                boxes.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'conf': conf,
                    'class': cls
                })
        return boxes

    def _nms_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """全局NMS去重"""
        if len(boxes) == 0:
            return boxes

        # 简单实现：按置信度排序，贪心去重
        boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
        keep = []

        for box in boxes:
            # 检查是否与已保留的框重叠过多
            is_duplicate = False
            for kept_box in keep:
                if self._iou(box, kept_box) > Config.YOLO_NMS_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(box)

        return keep

    def _iou(self, box1: Dict, box2: Dict) -> float:
        """计算两个框的IoU"""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

    def run(self):
        self._load_model()
        if not Config.PERFORMANCE_MODE:
            print("[Tier2-YOLO11m] 线程启动")

        while not stop_event.is_set():
            try:
                packet: FramePacket = tier2_queue.get(timeout=1.0)

                # 双模型推理（返回详细时间分解）
                boxes, model_type, timing = self._inference(packet.image, packet.roi_boxes)
                max_conf = max([b['conf'] for b in boxes]) if boxes else 0.0

                # 决定source标签（根据实际使用的模型）
                if model_type == "roi":
                    source = "tier2_roi"  # PyTorch ROI模型
                elif model_type == "full":
                    source = "tier2"      # TensorRT全图模型
                else:
                    source = "tier2_sim"  # 模拟模式

                # 输出结果
                result = DetectionResult(
                    frame_id=packet.frame_id,
                    boxes=boxes,
                    source=source,
                    latency_ms=timing.total_ms,
                    timestamp=packet.timestamp,
                    timing=timing,
                    max_conf=max_conf,
                    num_boxes=len(boxes)
                )
                result_queue.put(result)
                
                # 🆕 训练模式：同时放入训练结果队列
                if Config.TRAIN_MODE:
                    train_result_queue.put(result)

                if Config.ENABLE_STATS and not Config.PERFORMANCE_MODE:
                    model_info = "ROI(320)" if model_type == "roi" else "Full(640)"
                    roi_count = len(packet.roi_boxes) if packet.roi_boxes else 0
                    print(f"[Tier2] Frame {packet.frame_id}: {model_info}, ROI:{roi_count}, "
                          f"total={timing.total_ms:.2f}ms (pre={timing.preprocess_ms:.1f}, "
                          f"inf={timing.inference_ms:.1f}, post={timing.postprocess_ms:.1f}), "
                          f"检测:{len(boxes)}个")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Tier2-YOLO11m] 错误: {e}")
                import traceback
                traceback.print_exc()

        print("[Tier2-YOLO11m] 线程结束")


# ==================== 结果处理线程 ====================

class ResultProcessor(threading.Thread):
    """
    结果处理与显示
    职责：
    1. 从 result_queue 取结果
    2. 统计性能指标（重点：能效比）
    3. 绘制检测框（可选）
    """
    def __init__(self):
        super().__init__(daemon=True, name="ResultProcessor")
        self.stats = defaultdict(int)
        self.latencies = defaultdict(list)
        self.start_time = None
        self.total_frames = 0

        # 检测精度统计
        self.detection_stats = defaultdict(list)  # 每层检测框数量统计
        self.total_detections = 0  # 总检测框数量

        # 详细时间分解统计
        self.timing_stats = defaultdict(lambda: {
            'preprocess': [], 'inference': [], 'postprocess': [], 'total': []
        })

        # 每帧详细记录（用于绘图）
        self.frame_logs = []
        
        # 🆕 MOT评估：存储每帧的检测结果
        self.detection_results_by_frame: Dict[int, List[Dict]] = {}
        
        # 🆕 初始化MOT评估器
        self.mot_evaluator = None
        if Config.ENABLE_MOT_EVAL and MOT_EVALUATOR_AVAILABLE:
            try:
                eval_config = MOTEvalConfig(
                    iou_threshold=Config.MOT_IOU_THRESHOLD,
                    target_class=Config.MOT_TARGET_CLASS,
                    original_width=640,
                    original_height=640,
                    target_size=640
                )
                self.mot_evaluator = MOTEvaluator(Config.GT_FILE, eval_config)
                print(f"[ResultProcessor] MOT评估器初始化成功")
            except Exception as e:
                print(f"[ResultProcessor] ⚠️ MOT评估器初始化失败: {e}")
                import traceback
                traceback.print_exc()
                self.mot_evaluator = None
        elif Config.ENABLE_MOT_EVAL and not MOT_EVALUATOR_AVAILABLE:
            print(f"[ResultProcessor] ⚠️ MOT评估已启用但模块不可用")
        
        # 🆕 GMC评估：追踪预测准确性
        self.gmc_evaluations = []  # [(predicted_boxes, actual_boxes, iou_scores)]
        self.motion_vectors = []   # [(dx, dy, frame_id)] 运动向量记录
        self.source_transitions = defaultdict(lambda: defaultdict(int))  # {source: {next_source: count}}
        self.last_source = None    # 上一帧的source
        self.last_boxes_by_source: Dict[str, List[Dict]] = {}  # 每个source的最后检测框
        self.gmc_box_ious = []     # GMC预测框与GT的IoU列表
        
        # 🆕 额外GMC统计
        self.gmc_frame_count = 0           # GMC帧数
        self.tier0_predict_frame_count = 0 # 纯卡尔曼预测帧数
        self.gmc_gt_evaluations = []       # GMC与GT的对比评估: [{frame_id, predicted, gt, ious}]
        self.gmc_accuracy_by_threshold = defaultdict(lambda: {'correct': 0, 'total': 0})  # 不同阈值下的准确率

    def run(self):
        if not Config.PERFORMANCE_MODE:
            print("[ResultProcessor] 线程启动")
        self.start_time = time.time()

        while not stop_event.is_set():
            try:
                result: DetectionResult = result_queue.get(timeout=1.0)

                # 统计
                self.stats[result.source] += 1
                self.latencies[result.source].append(result.latency_ms)
                self.total_frames += 1

                # 检测框统计
                num_boxes = len(result.boxes)
                self.detection_stats[result.source].append(num_boxes)
                self.total_detections += num_boxes
                
                # 🆕 更新共享检测框（供Tier0运动预测使用）
                # 只有推理层的结果才更新，避免用预测框更新
                if result.source in ("tier1", "tier2", "tier2_roi") and num_boxes > 0:
                    global latest_detection_boxes, latest_detection_frame_id, latest_detection_lock
                    with latest_detection_lock:
                        latest_detection_boxes = [box.copy() for box in result.boxes]  # 深拷贝
                        latest_detection_frame_id = result.frame_id
                    
                    # 🆕 更新卡尔曼跟踪器（用检测结果更新跟踪状态）
                    global global_kalman_tracker, kalman_tracker_lock
                    if global_kalman_tracker is not None:
                        with kalman_tracker_lock:
                            # 使用检测结果更新跟踪器
                            global_kalman_tracker.update(result.boxes)
                
                # 🆕 GMC评估：收集数据
                # 1. 记录源转换（分析层级调度效果）
                if self.last_source is not None:
                    self.source_transitions[self.last_source][result.source] += 1
                self.last_source = result.source
                
                # 🆕 记录GMC/卡尔曼帧数
                if result.source == "tier0_gmc":
                    self.gmc_frame_count += 1
                elif result.source == "tier0_predict":
                    self.tier0_predict_frame_count += 1
                
                # 2. 评估GMC预测准确性（与实际推理结果对比）
                # 当GMC预测帧之后出现推理帧时，计算预测准确性
                if result.source in ("tier1", "tier2", "tier2_roi"):
                    # 检查上一帧是否是GMC预测
                    gmc_source = "tier0_gmc"
                    if gmc_source in self.last_boxes_by_source:
                        predicted_boxes = self.last_boxes_by_source[gmc_source]
                        actual_boxes = result.boxes
                        if predicted_boxes and actual_boxes:
                            # 计算预测框与实际框的IoU
                            ious = self._calculate_box_ious(predicted_boxes, actual_boxes)
                            self.gmc_evaluations.append({
                                'frame_id': result.frame_id,
                                'predicted_count': len(predicted_boxes),
                                'actual_count': len(actual_boxes),
                                'ious': ious,
                                'mean_iou': np.mean(ious) if ious else 0.0
                            })
                
                # 3. GMC与GT直接对比评估（如果有MOT评估器）
                if result.source == "tier0_gmc" and self.mot_evaluator is not None:
                    mot_frame_id = result.frame_id + 1
                    gt_boxes = self.mot_evaluator.gt_by_frame.get(mot_frame_id, [])
                    if gt_boxes and result.boxes:
                        # 计算预测框与GT的IoU
                        pred_boxes_list = result.boxes
                        gt_boxes_list = [{'x1': gt['x1'], 'y1': gt['y1'], 
                                         'x2': gt['x2'], 'y2': gt['y2']} for gt in gt_boxes]
                        ious = self._calculate_box_ious(pred_boxes_list, gt_boxes_list)
                        
                        self.gmc_gt_evaluations.append({
                            'frame_id': mot_frame_id,
                            'predicted_count': len(pred_boxes_list),
                            'gt_count': len(gt_boxes),
                            'ious': ious,
                            'mean_iou': np.mean(ious) if ious else 0.0
                        })
                        
                        # 更新各阈值下的准确率
                        for thresh in [0.3, 0.5, 0.7]:
                            for iou in ious:
                                self.gmc_accuracy_by_threshold[thresh]['total'] += 1
                                if iou >= thresh:
                                    self.gmc_accuracy_by_threshold[thresh]['correct'] += 1
                
                # 4. 存储每个source的最后检测框
                if num_boxes > 0:
                    self.last_boxes_by_source[result.source] = [box.copy() for box in result.boxes]
                
                # 🆕 存储检测结果用于MOT评估
                # 注意：frame_id从0开始，MOT GT从1开始，需要+1
                mot_frame_id = result.frame_id + 1
                self.detection_results_by_frame[mot_frame_id] = result.boxes

                # 时间分解统计
                if result.timing:
                    timing_stat = self.timing_stats[result.source]
                    timing_stat['preprocess'].append(result.timing.preprocess_ms)
                    timing_stat['inference'].append(result.timing.inference_ms)
                    timing_stat['postprocess'].append(result.timing.postprocess_ms)
                    timing_stat['total'].append(result.timing.total_ms)

                # 记录每帧详细数据（用于后续绘图）
                frame_log = {
                    'frame_id': result.frame_id,
                    'timestamp': result.timestamp,
                    'source': result.source,
                    'num_boxes': num_boxes,
                    'max_conf': result.max_conf,
                    'latency_ms': result.latency_ms,
                }
                if result.timing:
                    frame_log['timing'] = {
                        'preprocess_ms': result.timing.preprocess_ms,
                        'inference_ms': result.timing.inference_ms,
                        'postprocess_ms': result.timing.postprocess_ms,
                        'total_ms': result.timing.total_ms,
                    }
                self.frame_logs.append(frame_log)

                # 定期打印进度（静默模式下不打印）
                if Config.ENABLE_STATS and not Config.PERFORMANCE_MODE and self.total_frames % Config.STATS_PRINT_INTERVAL == 0:
                    self._print_progress()

                # 显示（可选）
                if Config.DISPLAY_OUTPUT:
                    self._display_result(result)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ResultProcessor] 错误: {e}")

        # 打印最终统计
        self._print_final_stats()
        
        # 🆕 执行MOT评估
        if self.mot_evaluator is not None:
            self._run_mot_evaluation()
        
        # 🆕 训练模式：保存训练数据
        if Config.TRAIN_MODE and global_train_collector is not None:
            global_train_collector.save_to_file(Config.TRAIN_DATA_OUTPUT)
        
        if not Config.PERFORMANCE_MODE:
            print("[ResultProcessor] 线程结束")

    def _print_progress(self):
        """打印实时进度"""
        elapsed = time.time() - self.start_time
        fps = self.total_frames / elapsed if elapsed > 0 else 0
        early_exit_count = self.stats.get("tier0_predict", 0) + self.stats.get("tier0_gmc", 0)
        early_exit_ratio = early_exit_count / self.total_frames * 100 if self.total_frames > 0 else 0

        print(f"[Progress] 帧数: {self.total_frames}, FPS: {fps:.1f}, "
              f"早退率: {early_exit_ratio:.1f}%, 队列: {result_queue.qsize()}")

    def _display_result(self, result: DetectionResult):
        """显示检测结果"""
        if result.image is None:
            # 如果没有图像，则跳过显示
            return

        # 复制图像（避免修改原图）
        display_img = result.image.copy()

        # 根据来源选择颜色
        color_map = {
            "tier0_predict": (128, 128, 128),  # 灰色 - 预测
            "tier0_gmc": (255, 165, 0),        # 橙色 - GMC预测
            "tier1": (0, 255, 0),              # 绿色 - Tier1
            "tier2": (0, 0, 255),              # 红色 - Tier2全图
            "tier2_roi": (255, 0, 255),        # 紫色 - Tier2 ROI
        }
        color = color_map.get(result.source, (255, 255, 255))

        # 绘制检测框
        for box in result.boxes:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            conf = box['conf']
            cls = box['class']

            # 绘制矩形框
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"cls:{cls} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_img, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(display_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示帧信息
        info_text = f"Frame:{result.frame_id} Source:{result.source} Latency:{result.latency_ms:.1f}ms"
        cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 显示图像
        cv2.imshow("Detection Results", display_img)
        cv2.waitKey(1)  # 1ms延迟，允许窗口更新

    def _calculate_single_iou(self, box1: Dict, box2: Dict) -> float:
        """计算两个框的IoU"""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return float(iou)

    def _calculate_box_ious(self, predicted_boxes: List[Dict], actual_boxes: List[Dict]) -> List[float]:
        """
        计算预测框与实际框的IoU列表
        使用贪婪匹配：每个预测框匹配IoU最高的实际框
        """
        if not predicted_boxes or not actual_boxes:
            return []
        
        ious = []
        used_actual = set()
        
        for pred_box in predicted_boxes:
            best_iou = 0.0
            best_idx = -1
            
            for idx, actual_box in enumerate(actual_boxes):
                if idx in used_actual:
                    continue
                iou = self._calculate_single_iou(pred_box, actual_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            ious.append(best_iou)
            if best_idx >= 0:
                used_actual.add(best_idx)
        
        return ious

    def _estimate_power(self):
        """估算平均功耗"""
        total = sum(self.stats.values())
        if total == 0:
            return 0.0

        # 计算各层级占比
        tier0_ratio = (self.stats.get("tier0_predict", 0) + self.stats.get("tier0_gmc", 0)) / total
        tier1_ratio = self.stats.get("tier1", 0) / total
        tier2_ratio = (self.stats.get("tier2", 0) + self.stats.get("tier2_roi", 0)) / total

        # 加权平均功耗
        avg_power = (
            tier0_ratio * Config.POWER_LSTM +
            tier1_ratio * Config.POWER_YOLO11N +
            tier2_ratio * Config.POWER_YOLO11M
        )

        return avg_power

    def _print_final_stats(self):
        """打印最终性能统计（重点：能效比）"""
        print("\n" + "="*70)
        print("📊 最终性能统计报告（能效优化为核心目标）")
        print("="*70)

        total_frames = sum(self.stats.values())
        elapsed_time = time.time() - self.start_time if self.start_time else 1
        fps = total_frames / elapsed_time

        # ========== 1. 基本性能 ==========
        print(f"\n【基本性能】")
        print(f"  总帧数:       {total_frames}")
        print(f"  总耗时:       {elapsed_time:.2f} s")
        print(f"  端到端FPS:    {fps:.2f}")

        # ========== 2. 早退统计（核心指标）==========
        print(f"\n【早退统计】（核心优化目标）")
        tier0_predict_count = self.stats.get("tier0_predict", 0)
        tier0_gmc_count = self.stats.get("tier0_gmc", 0)
        early_exit_count = tier0_predict_count + tier0_gmc_count
        tier1_count = self.stats.get("tier1", 0)
        tier2_count = self.stats.get("tier2", 0) + self.stats.get("tier2_roi", 0)
        inference_count = tier1_count + tier2_count
        early_exit_ratio = early_exit_count / total_frames * 100 if total_frames > 0 else 0

        print(f"  早退帧数:     {early_exit_count} ({early_exit_ratio:.1f}%)")
        print(f"    - 卡尔曼预测: {tier0_predict_count} 帧")
        print(f"    - GMC预测:    {tier0_gmc_count} 帧 ⭐")
        print(f"  推理帧数:     {inference_count} ({100-early_exit_ratio:.1f}%)")
        print(f"    - Tier1:      {tier1_count} 帧")
        print(f"    - Tier2:      {tier2_count} 帧")
        print(f"  目标早退率:   {Config.TARGET_EARLY_EXIT_RATIO * 100:.1f}%")

        if early_exit_ratio >= Config.TARGET_EARLY_EXIT_RATIO * 100:
            print(f"  ✅ 达成早退目标！")
        else:
            print(f"  ⚠️  未达标，建议调整 LSTM_THRESHOLD_HIGH")

        # ========== 3. 各层级分布 ==========
        print(f"\n【层级分布】")
        for source, count in sorted(self.stats.items()):
            ratio = count / total_frames * 100 if total_frames > 0 else 0
            avg_latency = np.mean(self.latencies[source]) if self.latencies[source] else 0
            print(f"  {source:20s}: {count:5d} ({ratio:5.1f}%)  延迟: {avg_latency:6.2f}ms")

        # ========== 3.5 详细模型性能分析 ==========
        print(f"\n【详细模型性能分析】")

        # Tier1 (YOLO11n)
        tier1_latencies = self.latencies.get("tier1", [])
        if tier1_latencies:
            print(f"\n  📊 Tier1 (YOLO11n TensorRT 640×640):")
            print(f"     帧数:       {len(tier1_latencies)}")
            print(f"     平均延迟:   {np.mean(tier1_latencies):.2f} ms")
            print(f"     最小延迟:   {np.min(tier1_latencies):.2f} ms")
            print(f"     最大延迟:   {np.max(tier1_latencies):.2f} ms")
            print(f"     标准差:     {np.std(tier1_latencies):.2f} ms")
            print(f"     中位数:     {np.median(tier1_latencies):.2f} ms")
            print(f"     P95延迟:    {np.percentile(tier1_latencies, 95):.2f} ms")
            print(f"     P99延迟:    {np.percentile(tier1_latencies, 99):.2f} ms")

            # 🔥 时间分解
            if "tier1" in self.timing_stats:
                ts = self.timing_stats["tier1"]
                if ts['preprocess']:
                    print(f"     --- 时间分解 ---")
                    print(f"     前处理:     {np.mean(ts['preprocess']):.2f} ms (±{np.std(ts['preprocess']):.2f})")
                    print(f"     GPU推理:    {np.mean(ts['inference']):.2f} ms (±{np.std(ts['inference']):.2f})")
                    print(f"     后处理:     {np.mean(ts['postprocess']):.2f} ms (±{np.std(ts['postprocess']):.2f})")

        # Tier2 Full (YOLO11m TensorRT)
        tier2_full_latencies = self.latencies.get("tier2", [])
        if tier2_full_latencies:
            print(f"\n  📊 Tier2-Full (YOLO11m TensorRT 640×640):")
            print(f"     帧数:       {len(tier2_full_latencies)}")
            print(f"     平均延迟:   {np.mean(tier2_full_latencies):.2f} ms")
            print(f"     最小延迟:   {np.min(tier2_full_latencies):.2f} ms")
            print(f"     最大延迟:   {np.max(tier2_full_latencies):.2f} ms")
            print(f"     标准差:     {np.std(tier2_full_latencies):.2f} ms")
            print(f"     中位数:     {np.median(tier2_full_latencies):.2f} ms")
            print(f"     P95延迟:    {np.percentile(tier2_full_latencies, 95):.2f} ms")
            print(f"     P99延迟:    {np.percentile(tier2_full_latencies, 99):.2f} ms")

            # 🔥 时间分解
            if "tier2" in self.timing_stats:
                ts = self.timing_stats["tier2"]
                if ts['preprocess']:
                    print(f"     --- 时间分解 ---")
                    print(f"     前处理:     {np.mean(ts['preprocess']):.2f} ms (±{np.std(ts['preprocess']):.2f})")
                    print(f"     GPU推理:    {np.mean(ts['inference']):.2f} ms (±{np.std(ts['inference']):.2f})")
                    print(f"     后处理:     {np.mean(ts['postprocess']):.2f} ms (±{np.std(ts['postprocess']):.2f})")

        # Tier2 ROI (YOLO11m TensorRT/PyTorch)
        tier2_roi_latencies = self.latencies.get("tier2_roi", [])
        if tier2_roi_latencies:
            # 检测模型类型
            model_type = "TensorRT" if Config.MODEL_YOLO11M_ROI.endswith('.engine') else "PyTorch"
            print(f"\n  📊 Tier2-ROI (YOLO11m {model_type} 320×320):")
            print(f"     帧数:       {len(tier2_roi_latencies)}")
            print(f"     平均延迟:   {np.mean(tier2_roi_latencies):.2f} ms")
            print(f"     最小延迟:   {np.min(tier2_roi_latencies):.2f} ms")
            print(f"     最大延迟:   {np.max(tier2_roi_latencies):.2f} ms")
            print(f"     标准差:     {np.std(tier2_roi_latencies):.2f} ms")
            print(f"     中位数:     {np.median(tier2_roi_latencies):.2f} ms")
            print(f"     P95延迟:    {np.percentile(tier2_roi_latencies, 95):.2f} ms")
            print(f"     P99延迟:    {np.percentile(tier2_roi_latencies, 99):.2f} ms")

            # 🔥 时间分解
            if "tier2_roi" in self.timing_stats:
                ts = self.timing_stats["tier2_roi"]
                if ts['preprocess']:
                    print(f"     --- 时间分解 ---")
                    print(f"     前处理:     {np.mean(ts['preprocess']):.2f} ms (±{np.std(ts['preprocess']):.2f})")
                    print(f"     GPU推理:    {np.mean(ts['inference']):.2f} ms (±{np.std(ts['inference']):.2f})")
                    print(f"     后处理:     {np.mean(ts['postprocess']):.2f} ms (±{np.std(ts['postprocess']):.2f})")

            # ROI vs Full对比
            if tier2_full_latencies:
                speedup = np.mean(tier2_full_latencies) / np.mean(tier2_roi_latencies)
                print(f"\n  🔥 ROI加速比: {speedup:.2f}x ({np.mean(tier2_full_latencies):.2f}ms → {np.mean(tier2_roi_latencies):.2f}ms)")

        # ========== 3.6 检测结果统计 ==========
        print(f"\n【检测结果统计】")
        print(f"  总检测框数:    {self.total_detections}")
        print(f"  平均每帧检测:  {self.total_detections / total_frames if total_frames > 0 else 0:.2f} 个")

        for source in sorted(self.detection_stats.keys()):
            detections = self.detection_stats[source]
            if detections:
                avg_boxes = np.mean(detections)
                max_boxes = np.max(detections)
                min_boxes = np.min(detections)
                print(f"\n  📦 {source}:")
                print(f"     帧数:       {len(detections)}")
                print(f"     总检测框:   {sum(detections)}")
                print(f"     平均检测:   {avg_boxes:.2f} 个/帧")
                print(f"     最小/最大:  {min_boxes} / {max_boxes} 个")

        # ========== 4. 能效分析（论文核心指标）==========
        print(f"\n【能效分析】（论文核心贡献）")
        estimated_power = self._estimate_power()
        baseline_power = Config.POWER_YOLO11M  # Baseline: 全量YOLO11m
        energy_efficiency = fps / estimated_power if estimated_power > 0 else 0
        baseline_efficiency = fps / baseline_power

        print(f"  估算功耗:     {estimated_power:.2f} W")
        print(f"  Baseline功耗: {baseline_power:.2f} W")
        print(f"  功耗降低:     {(1 - estimated_power/baseline_power)*100:.1f}%")
        print(f"  能效比:       {energy_efficiency:.2f} FPS/W")
        print(f"  Baseline能效: {baseline_efficiency:.2f} FPS/W")
        print(f"  能效提升:     {(energy_efficiency/baseline_efficiency - 1)*100:.1f}%")

        if estimated_power <= Config.TARGET_POWER_BUDGET:
            print(f"  ✅ 功耗达标！（目标 {Config.TARGET_POWER_BUDGET}W）")
        else:
            print(f"  ⚠️  功耗超标，建议提高早退率")

        # ========== 5. GMC预测评估 ==========
        self._print_gmc_evaluation()

        # ========== 6. 层级转换分析 ==========
        self._print_source_transitions()

        # ========== 7. 🆕 总结信息（重点显示GMC帧数）==========
        print(f"\n" + "="*70)
        print(f"📋 【关键统计总结】")
        print(f"="*70)
        print(f"  总帧数:          {total_frames}")
        print(f"  处理FPS:         {fps:.2f}")
        print(f"  早退率:          {early_exit_ratio:.1f}%")
        print(f"")
        print(f"  🌟 GMC帧数:       {tier0_gmc_count} 帧 ({tier0_gmc_count/total_frames*100:.1f}%)")
        print(f"  🌟 卡尔曼帧数:    {tier0_predict_count} 帧 ({tier0_predict_count/total_frames*100:.1f}%)")
        print(f"  🌟 Tier1帧数:     {tier1_count} 帧 ({tier1_count/total_frames*100:.1f}%)")
        print(f"  🌟 Tier2帧数:     {tier2_count} 帧 ({tier2_count/total_frames*100:.1f}%)")
        print(f"")
        print(f"  估算功耗:        {estimated_power:.2f} W")
        print(f"  能效比:          {energy_efficiency:.2f} FPS/W")
        print(f"="*70)

        # ========== 8. 保存日志（可选）==========
        if Config.SAVE_LOGS:
            self._save_logs(fps, early_exit_ratio, estimated_power, energy_efficiency)

    def _print_gmc_evaluation(self):
        """打印GMC预测评估结果"""
        print(f"\n【GMC预测评估】（运动补偿预测准确性）")
        print(f"  GMC帧数:       {self.gmc_frame_count} 帧")
        print(f"  卡尔曼帧数:    {self.tier0_predict_frame_count} 帧")
        
        # 如果有GMC与GT的评估但没有GMC与推理的评估，也要显示
        if not self.gmc_evaluations and not self.gmc_gt_evaluations:
            print("  ⚠️  没有GMC评估数据（可能没有GMC预测帧）")
            return
        
        if not self.gmc_evaluations:
            print("  ⚠️  没有GMC与下一帧推理的对比数据（GMC帧后未跟随推理帧）")
            # 继续显示GMC与GT的对比（如果有的话）
            if self.gmc_gt_evaluations:
                self._print_gmc_gt_only()
            return
        
        # 汇总统计
        all_ious = []
        total_predicted = 0
        total_actual = 0
        
        for eval_data in self.gmc_evaluations:
            all_ious.extend(eval_data['ious'])
            total_predicted += eval_data['predicted_count']
            total_actual += eval_data['actual_count']
        
        if not all_ious:
            print("  ⚠️  没有可匹配的框（预测或实际检测为空）")
            return
        
        # IoU统计
        mean_iou = np.mean(all_ious)
        median_iou = np.median(all_ious)
        std_iou = np.std(all_ious)
        min_iou = np.min(all_ious)
        max_iou = np.max(all_ious)
        
        print(f"  📈 与下一帧推理结果对比：")
        print(f"     评估帧数:    {len(self.gmc_evaluations)}")
        print(f"     预测框总数:  {total_predicted}")
        print(f"     实际框总数:  {total_actual}")
        print(f"")
        print(f"     📊 IoU统计:")
        print(f"        平均IoU:    {mean_iou:.4f}")
        print(f"        中位数IoU:  {median_iou:.4f}")
        print(f"        标准差:     {std_iou:.4f}")
        print(f"        最小/最大:  {min_iou:.4f} / {max_iou:.4f}")
        
        # 不同阈值下的成功率
        thresholds = [0.3, 0.5, 0.7]
        print(f"")
        print(f"     🎯 预测成功率（IoU >= 阈值）:")
        for thresh in thresholds:
            success_count = sum(1 for iou in all_ious if iou >= thresh)
            success_rate = success_count / len(all_ious) * 100
            print(f"        IoU >= {thresh}: {success_count}/{len(all_ious)} ({success_rate:.1f}%)")
        
        # 完全失败的预测（IoU = 0）
        zero_iou_count = sum(1 for iou in all_ious if iou == 0)
        zero_iou_rate = zero_iou_count / len(all_ious) * 100
        print(f"")
        print(f"     ⚠️  完全失败预测（IoU=0）: {zero_iou_count} ({zero_iou_rate:.1f}%)")
        
        # ========== 🆕 GMC与GT直接对比 ==========
        if self.gmc_gt_evaluations:
            print(f"")
            print(f"  📈 与Ground Truth对比（GMC vs GT）: ⭐核心指标")
            
            gt_all_ious = []
            total_gt_predicted = 0
            total_gt = 0
            
            for eval_data in self.gmc_gt_evaluations:
                gt_all_ious.extend(eval_data['ious'])
                total_gt_predicted += eval_data['predicted_count']
                total_gt += eval_data['gt_count']
            
            if gt_all_ious:
                gt_mean_iou = np.mean(gt_all_ious)
                gt_median_iou = np.median(gt_all_ious)
                gt_std_iou = np.std(gt_all_ious)
                
                print(f"     GMC帧数:     {len(self.gmc_gt_evaluations)}")
                print(f"     预测框总数:  {total_gt_predicted}")
                print(f"     GT框总数:    {total_gt}")
                print(f"")
                print(f"     📊 IoU统计（与GT）:")
                print(f"        平均IoU:    {gt_mean_iou:.4f}")
                print(f"        中位数IoU:  {gt_median_iou:.4f}")
                print(f"        标准差:     {gt_std_iou:.4f}")
                
                # 各阈值下的准确率
                print(f"")
                print(f"     🎯 GMC准确率（与GT对比）:")
                for thresh in [0.3, 0.5, 0.7]:
                    stats = self.gmc_accuracy_by_threshold[thresh]
                    if stats['total'] > 0:
                        accuracy = stats['correct'] / stats['total'] * 100
                        print(f"        IoU >= {thresh}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
                    else:
                        print(f"        IoU >= {thresh}: N/A")
        else:
            print(f"")
            print(f"  ⚠️  没有GMC与GT的对比数据（可能MOT评估器未初始化或无GMC帧）")
        
        # 评估结论
        print(f"")
        if mean_iou >= 0.5:
            print(f"  ✅ GMC预测质量良好（平均IoU >= 0.5）")
        elif mean_iou >= 0.3:
            print(f"  ⚠️  GMC预测质量中等（0.3 <= 平均IoU < 0.5）")
        else:
            print(f"  ❌ GMC预测质量较差（平均IoU < 0.3），建议调整运动估计参数")

    def _print_source_transitions(self):
        """打印层级转换分析"""
        print(f"\n【层级转换分析】（调度策略效果）")
        
        if not self.source_transitions:
            print("  ⚠️  没有转换数据")
            return
        
        # 收集所有出现过的source
        all_sources = set()
        for src, transitions in self.source_transitions.items():
            all_sources.add(src)
            all_sources.update(transitions.keys())
        all_sources = sorted(all_sources)
        
        # 打印转换矩阵
        print(f"\n  📊 转换矩阵（当前帧 → 下一帧）:")
        
        # 表头
        header = "  " + " " * 18 + "".join(f"{s[:10]:>12}" for s in all_sources)
        print(header)
        print("  " + "-" * (18 + 12 * len(all_sources)))
        
        for src in all_sources:
            row = f"  {src:16s} |"
            for dst in all_sources:
                count = self.source_transitions.get(src, {}).get(dst, 0)
                row += f"{count:>12}"
            print(row)
        
        # 分析关键转换
        print(f"\n  🔍 关键转换分析:")
        
        # GMC预测后的转换分布
        gmc_source = "tier0_gmc"
        if gmc_source in self.source_transitions:
            gmc_transitions = self.source_transitions[gmc_source]
            gmc_total = sum(gmc_transitions.values())
            print(f"\n     GMC预测后的转换（共{gmc_total}次）:")
            for dst, count in sorted(gmc_transitions.items(), key=lambda x: -x[1]):
                ratio = count / gmc_total * 100 if gmc_total > 0 else 0
                print(f"       → {dst}: {count} ({ratio:.1f}%)")
            
            # 连续GMC预测能力
            consecutive_gmc = gmc_transitions.get(gmc_source, 0)
            if gmc_total > 0:
                consecutive_ratio = consecutive_gmc / gmc_total * 100
                print(f"\n     连续GMC预测率: {consecutive_ratio:.1f}%")
        
        # 推理层后的转换分布
        inference_sources = ["tier1", "tier2", "tier2_roi"]
        for inf_src in inference_sources:
            if inf_src in self.source_transitions:
                transitions = self.source_transitions[inf_src]
                total = sum(transitions.values())
                early_exit = transitions.get("tier0_predict", 0) + transitions.get("tier0_gmc", 0)
                early_ratio = early_exit / total * 100 if total > 0 else 0
                print(f"\n     {inf_src}后早退率: {early_ratio:.1f}% ({early_exit}/{total})")

    def _print_gmc_gt_only(self):
        """仅打印GMC与GT的对比评估（当没有GMC与推理对比时使用）"""
        print(f"")
        print(f"  📈 与Ground Truth对比（GMC vs GT）: ⭐核心指标")
        
        gt_all_ious = []
        total_gt_predicted = 0
        total_gt = 0
        
        for eval_data in self.gmc_gt_evaluations:
            gt_all_ious.extend(eval_data['ious'])
            total_gt_predicted += eval_data['predicted_count']
            total_gt += eval_data['gt_count']
        
        if gt_all_ious:
            gt_mean_iou = np.mean(gt_all_ious)
            gt_median_iou = np.median(gt_all_ious)
            gt_std_iou = np.std(gt_all_ious)
            
            print(f"     GMC帧数:     {len(self.gmc_gt_evaluations)}")
            print(f"     预测框总数:  {total_gt_predicted}")
            print(f"     GT框总数:    {total_gt}")
            print(f"")
            print(f"     📊 IoU统计（与GT）:")
            print(f"        平均IoU:    {gt_mean_iou:.4f}")
            print(f"        中位数IoU:  {gt_median_iou:.4f}")
            print(f"        标准差:     {gt_std_iou:.4f}")
            
            # 各阈值下的准确率
            print(f"")
            print(f"     🎯 GMC准确率（与GT对比）:")
            for thresh in [0.3, 0.5, 0.7]:
                stats = self.gmc_accuracy_by_threshold[thresh]
                if stats['total'] > 0:
                    accuracy = stats['correct'] / stats['total'] * 100
                    print(f"        IoU >= {thresh}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
                else:
                    print(f"        IoU >= {thresh}: N/A")
            
            # 评估结论
            print(f"")
            if gt_mean_iou >= 0.5:
                print(f"  ✅ GMC预测质量良好（平均IoU >= 0.5）")
            elif gt_mean_iou >= 0.3:
                print(f"  ⚠️  GMC预测质量中等（0.3 <= 平均IoU < 0.5）")
            else:
                print(f"  ❌ GMC预测质量较差（平均IoU < 0.3），建议调整运动估计参数")
        else:
            print(f"  ⚠️  没有可匹配的框")

    def _save_logs(self, fps, early_exit_ratio, estimated_power, energy_efficiency):
        """保存日志到文件（包含详细的每帧数据用于绘图）"""
        import json
        from datetime import datetime

        # 构建时间分解统计
        timing_stats_output = {}
        for source, ts in self.timing_stats.items():
            if ts['preprocess']:
                timing_stats_output[source] = {
                    'preprocess': {
                        'mean': float(np.mean(ts['preprocess'])),
                        'std': float(np.std(ts['preprocess'])),
                        'min': float(np.min(ts['preprocess'])),
                        'max': float(np.max(ts['preprocess'])),
                        'p95': float(np.percentile(ts['preprocess'], 95)),
                    },
                    'inference': {
                        'mean': float(np.mean(ts['inference'])),
                        'std': float(np.std(ts['inference'])),
                        'min': float(np.min(ts['inference'])),
                        'max': float(np.max(ts['inference'])),
                        'p95': float(np.percentile(ts['inference'], 95)),
                    },
                    'postprocess': {
                        'mean': float(np.mean(ts['postprocess'])),
                        'std': float(np.std(ts['postprocess'])),
                        'min': float(np.min(ts['postprocess'])),
                        'max': float(np.max(ts['postprocess'])),
                        'p95': float(np.percentile(ts['postprocess'], 95)),
                    },
                    'total': {
                        'mean': float(np.mean(ts['total'])),
                        'std': float(np.std(ts['total'])),
                        'min': float(np.min(ts['total'])),
                        'max': float(np.max(ts['total'])),
                        'p95': float(np.percentile(ts['total'], 95)),
                    }
                }

        # 构建延迟分布统计
        latency_stats = {}
        for source, latencies in self.latencies.items():
            if latencies:
                latency_stats[source] = {
                    'count': len(latencies),
                    'mean': float(np.mean(latencies)),
                    'std': float(np.std(latencies)),
                    'min': float(np.min(latencies)),
                    'max': float(np.max(latencies)),
                    'median': float(np.median(latencies)),
                    'p95': float(np.percentile(latencies, 95)),
                    'p99': float(np.percentile(latencies, 99)),
                }

        log_data = {
            # 元信息
            "timestamp": datetime.now().isoformat(),
            "video_path": Config.VIDEO_PATH,

            # 汇总统计
            "summary": {
                "total_frames": self.total_frames,
                "elapsed_time_s": time.time() - self.start_time if self.start_time else 0,
                "fps": fps,
                "early_exit_ratio": early_exit_ratio,
                "estimated_power_w": estimated_power,
                "energy_efficiency_fps_per_watt": energy_efficiency,
                "total_detections": self.total_detections,
                "gmc_frame_count": self.gmc_frame_count,  # 🆕 GMC帧数
                "tier0_predict_frame_count": self.tier0_predict_frame_count,  # 🆕 卡尔曼帧数
            },

            # 层级分布
            "tier_distribution": dict(self.stats),

            # 延迟统计（汇总）
            "latency_stats": latency_stats,

            # 🔥 时间分解统计（关键！）
            "timing_breakdown_stats": timing_stats_output,

            # 🆕 GMC评估统计
            "gmc_evaluation": {
                "gmc_frame_count": self.gmc_frame_count,
                "tier0_predict_frame_count": self.tier0_predict_frame_count,
                "gmc_vs_inference_count": len(self.gmc_evaluations),
                "gmc_vs_gt_count": len(self.gmc_gt_evaluations),
                "accuracy_by_threshold": {
                    str(thresh): {
                        "correct": self.gmc_accuracy_by_threshold[thresh]['correct'],
                        "total": self.gmc_accuracy_by_threshold[thresh]['total'],
                        "rate": self.gmc_accuracy_by_threshold[thresh]['correct'] / 
                               max(1, self.gmc_accuracy_by_threshold[thresh]['total'])
                    } for thresh in [0.3, 0.5, 0.7]
                },
            },

            # 配置参数
            "config": {
                "lstm_threshold_low": Config.LSTM_THRESHOLD_LOW,
                "lstm_threshold_high": Config.LSTM_THRESHOLD_HIGH,
                "target_early_exit_ratio": Config.TARGET_EARLY_EXIT_RATIO,
                "tier1_conf_threshold": Config.TIER1_CONF_THRESHOLD,
                "input_size_tier1": Config.INPUT_SIZE_TIER1,
                "input_size_tier2_full": Config.INPUT_SIZE_TIER2_FULL,
                "input_size_tier2_roi": Config.INPUT_SIZE_TIER2_ROI,
                "performance_mode": Config.PERFORMANCE_MODE,
                "test_mode_force_tier1": Config.TEST_MODE_FORCE_TIER1,  # 🆕 记录测试模式状态
            },

            # 🔥 每帧详细数据（用于绘图）
            "frame_logs": self.frame_logs,
        }

        log_file = f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n💾 日志已保存: {log_file}")
        print(f"   - 帧数: {len(self.frame_logs)}")
        print(f"   - 包含: 每帧时间分解, 延迟分布, 配置参数")

    def _run_mot_evaluation(self):
        """🆕 执行MOT评估（运行结束后调用）"""
        if self.mot_evaluator is None:
            print("\n⚠️  MOT评估器未初始化，跳过评估")
            return
            
        print("\n" + "="*70)
        print("🎯 开始 MOT 检测精度评估...")
        print("="*70)
        
        # 评估所有帧
        evaluated_frames = 0
        for frame_id, detections in self.detection_results_by_frame.items():
            self.mot_evaluator.evaluate_frame(frame_id, detections)
            evaluated_frames += 1
            
        print(f"✅ 已评估 {evaluated_frames} 帧")
        
        # 打印评估结果
        metrics = self.mot_evaluator.print_summary()
        
        # 保存每帧详细指标到JSON
        if Config.SAVE_LOGS:
            self._save_mot_metrics(metrics)
    
    def _save_mot_metrics(self, overall_metrics: Dict):
        """保存MOT评估结果到JSON文件"""
        import json
        from datetime import datetime
        
        # 获取每帧详细指标
        per_frame_metrics = self.mot_evaluator.get_per_frame_metrics()
        
        mot_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "gt_file": Config.GT_FILE,
                "iou_threshold": Config.MOT_IOU_THRESHOLD,
                "target_class": Config.MOT_TARGET_CLASS,
            },
            "overall_metrics": overall_metrics,
            "per_frame_metrics": per_frame_metrics,
        }
        
        mot_file = f"mot_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(mot_file, 'w') as f:
            json.dump(mot_data, f, indent=2)
        
        print(f"\n💾 MOT评估结果已保存: {mot_file}")


# ==================== 主线程：视频读取 ====================

class VideoReader:
    """视频读取器（单进程版本）"""
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None

    def start(self):
        """启动视频读取"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")

        print(f"📹 视频读取启动: {self.video_path}")
        frame_id = 0
        start_time = time.time()

        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("📹 视频读取完成")
                break

            # 创建帧数据包
            packet = FramePacket(
                frame_id=frame_id,
                image=frame,
                timestamp=time.time()
            )

            # 放入队列（阻塞等待）
            try:
                frame_queue.put(packet, timeout=2.0)
                frame_id += 1
            except queue.Full:
                print("⚠️  frame_queue 已满，丢帧！")

        # 计算总FPS
        total_time = time.time() - start_time
        fps = frame_id / total_time if total_time > 0 else 0
        print(f"📹 读取完成: {frame_id} 帧, {total_time:.2f}s, {fps:.2f} FPS")

        self.cap.release()


# ==================== 多进程版本：VideoReader + Tier0 ====================

def video_reader_process(video_path: str, frame_index_queue: MPQueue,
                        stop_event: MPEvent, ready_event: MPEvent):
    """
    视频读取进程（包含Tier0 LSTM预测）
    职责：
    1. 读取视频帧
    2. LSTM预测（判断是否早退）
    3. 写入共享内存
    4. 通过Queue发送索引号
    """
    print("[VideoReaderProcess] 进程启动")

    # 连接到共享内存
    shm_buffer = SharedMemoryFrameBuffer(
        buffer_size=Config.SHARED_MEMORY_BUFFER_SIZE,
        width=Config.FRAME_WIDTH,
        height=Config.FRAME_HEIGHT,
        channels=Config.FRAME_CHANNELS,
        create=False  # 连接到主进程创建的共享内存
    )

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VideoReaderProcess] ❌ 无法打开视频: {video_path}")
        return

    print(f"[VideoReaderProcess] 📹 视频打开成功: {video_path}")

    # LSTM预测器（简单实现）
    prev_frame_gray = None

    # 通知主进程已就绪
    ready_event.set()
    print("[VideoReaderProcess] ✅ 就绪")

    frame_id = 0
    start_time = time.time()
    slot_idx = 0  # Ring Buffer索引

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[VideoReaderProcess] 📹 视频读取完成")
                break

            # Tier0: LSTM预测
            t0 = time.perf_counter()

            # 简单帧差计算（作为LSTM的placeholder）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame_gray is None:
                lstm_score = 1.0  # 第一帧强制推理
                action_type = ActionType.INVOKE_TIER1.value
            else:
                diff = cv2.absdiff(prev_frame_gray, gray)
                lstm_score = float(np.mean(diff) / 255.0)

                # 决策
                if Config.TEST_MODE_FORCE_TIER1:
                    action_type = ActionType.INVOKE_TIER1.value
                elif lstm_score < Config.LSTM_THRESHOLD_LOW:
                    action_type = ActionType.SKIP_PREDICT.value
                elif lstm_score < Config.LSTM_THRESHOLD_HIGH:
                    action_type = ActionType.SKIP_GMC.value
                else:
                    action_type = ActionType.INVOKE_TIER1.value

            prev_frame_gray = gray
            lstm_latency = (time.perf_counter() - t0) * 1000

            # 写入共享内存
            shm_buffer.write_frame(
                slot_idx=slot_idx,
                frame=frame,
                frame_id=frame_id,
                timestamp=time.time(),
                lstm_score=lstm_score,
                action=action_type
            )

            # 发送索引号到队列（轻量级通知）
            try:
                frame_index_queue.put(slot_idx, timeout=2.0)
            except:
                print("[VideoReaderProcess] ⚠️  Queue已满，丢帧！")

            # 更新Ring Buffer索引
            slot_idx = (slot_idx + 1) % Config.SHARED_MEMORY_BUFFER_SIZE
            frame_id += 1

            if not Config.PERFORMANCE_MODE and frame_id % Config.STATS_PRINT_INTERVAL == 0:
                elapsed = time.time() - start_time
                fps = frame_id / elapsed if elapsed > 0 else 0
                print(f"[VideoReaderProcess] Frame {frame_id}: lstm_score={lstm_score:.3f}, "
                      f"FPS={fps:.1f}, queue_size={frame_index_queue.qsize()}")

    except Exception as e:
        print(f"[VideoReaderProcess] ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        shm_buffer.close()

        total_time = time.time() - start_time
        fps = frame_id / total_time if total_time > 0 else 0
        print(f"[VideoReaderProcess] 完成: {frame_id} 帧, {total_time:.2f}s, {fps:.2f} FPS")
        print("[VideoReaderProcess] 进程结束")


# ==================== 主进程：帧分发器（替代Tier0线程）====================

class FrameDispatcher(threading.Thread):
    """
    从共享内存读取帧，根据LSTM预测结果分发（多进程版本）
    替代原来的Tier0线程
    """
    def __init__(self, shm_buffer: SharedMemoryFrameBuffer,
                 frame_index_queue: MPQueue, stop_event_mp: MPEvent):
        super().__init__(daemon=True, name="FrameDispatcher")
        self.shm_buffer = shm_buffer
        self.frame_index_queue = frame_index_queue
        self.stop_event_mp = stop_event_mp

    def run(self):
        if not Config.PERFORMANCE_MODE:
            print("[FrameDispatcher] 线程启动")

        while not stop_event.is_set() and not self.stop_event_mp.is_set():
            try:
                # 从Queue获取帧索引号（轻量级）
                slot_idx = self.frame_index_queue.get(timeout=1.0)

                # 从共享内存读取帧
                frame, frame_id, timestamp, lstm_score, action = self.shm_buffer.read_frame(slot_idx)

                # 创建帧数据包
                packet = FramePacket(
                    frame_id=frame_id,
                    image=frame,
                    timestamp=timestamp,
                    lstm_score=lstm_score,
                    action=ActionType(action)
                )

                # 根据action分发
                if packet.action == ActionType.SKIP_PREDICT or packet.action == ActionType.SKIP_GMC:
                    # 不需要推理，直接生成预测结果
                    result = self._generate_prediction_result(packet)
                    result_queue.put(result)

                elif packet.action == ActionType.INVOKE_TIER1:
                    # 需要推理，放入 Tier1 队列
                    tier1_queue.put(packet)

                if Config.ENABLE_STATS and not Config.PERFORMANCE_MODE and frame_id % Config.STATS_PRINT_INTERVAL == 0:
                    print(f"[FrameDispatcher] Frame {frame_id}: score={lstm_score:.3f}, "
                          f"action={packet.action.name_str}, "
                          f"queue(t1/t2): {tier1_queue.qsize()}/{tier2_queue.qsize()}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FrameDispatcher] 错误: {e}")
                import traceback
                traceback.print_exc()

        if not Config.PERFORMANCE_MODE:
            print("[FrameDispatcher] 线程结束")

    def _generate_prediction_result(self, packet: FramePacket) -> DetectionResult:
        """生成预测结果（卡尔曼/GMC）"""
        source = "tier0_gmc" if packet.action == ActionType.SKIP_GMC else "tier0_predict"
        return DetectionResult(
            frame_id=packet.frame_id,
            boxes=[],  # 空检测（实际应该有预测框）
            source=source,
            latency_ms=0.5,  # 预测很快
            timestamp=packet.timestamp
        )


# ==================== 主程序入口 ====================

def main():
    """主程序"""
    print("="*70)
    print("🚀 LSTM 时序预测 + 混合精度边缘检测系统")
    print("="*70)
    print(f"🔧 调试信息: ENABLE_MULTIPROCESSING = {Config.ENABLE_MULTIPROCESSING}")

    if Config.ENABLE_MULTIPROCESSING:
        print("🔥 多进程模式启用（绕过GIL）")
        print("   即将调用 main_multiprocess()...")
        main_multiprocess()
    else:
        print("📌 单进程模式（传统多线程）")
        print("   即将调用 main_singleprocess()...")
        main_singleprocess()


def main_multiprocess():
    """多进程模式主函数"""
    print("="*70)
    print("🚀 多进程架构启动")
    print("="*70)
    print(f"ℹ️  主进程PID: {mp.current_process().pid}")

    # 0️⃣ 清理可能残留的共享内存（防止上次异常退出未清理）
    print("\n🧹 检查并清理残留共享内存...")
    try:
        stale_shm = shared_memory.SharedMemory(name="frame_buffer_shm")
        stale_shm.close()
        stale_shm.unlink()
        print("   ✅ 已清理残留的共享内存")
    except FileNotFoundError:
        print("   ✅ 无残留共享内存")
    except Exception as e:
        print(f"   ⚠️  清理警告: {e}")
    
    # 🆕 训练模式初始化
    global global_train_collector
    if Config.TRAIN_MODE:
        print("\n⭐ 训练模式已启用")
        if TRAIN_COLLECTOR_AVAILABLE and MOT_EVALUATOR_AVAILABLE:
            try:
                # 先创建 MOT 评估器获取 GT 数据
                eval_config = MOTEvalConfig(
                    iou_threshold=Config.MOT_IOU_THRESHOLD,
                    target_class=Config.MOT_TARGET_CLASS,
                    original_width=640,
                    original_height=640,
                    target_size=640
                )
                temp_evaluator = MOTEvaluator(Config.GT_FILE, eval_config)
                
                # 创建训练数据收集器
                global_train_collector = TrainDataCollector(
                    gt_by_frame=temp_evaluator.gt_by_frame,
                    iou_threshold=Config.MOT_IOU_THRESHOLD
                )
                print("[Main] ✅ 训练数据收集器初始化成功")
            except Exception as e:
                print(f"[Main] ⚠️ 训练数据收集器初始化失败: {e}")
                import traceback
                traceback.print_exc()
                global_train_collector = None
        else:
            print("[Main] ⚠️ 训练模式需要 TrainDataCollector 和 MOTEvaluator 模块")
            global_train_collector = None


    # 1️⃣ 创建共享内存
    print("\n📦 第1步：创建共享内存...")
    shm_buffer = SharedMemoryFrameBuffer(
        buffer_size=Config.SHARED_MEMORY_BUFFER_SIZE,
        width=Config.FRAME_WIDTH,
        height=Config.FRAME_HEIGHT,
        channels=Config.FRAME_CHANNELS,
        create=True
    )

    # 2️⃣ 创建进程间通信
    print("📦 第2步：创建进程间通信...")
    frame_index_queue = MPQueue(maxsize=Config.SHARED_MEMORY_BUFFER_SIZE)
    stop_event_mp = MPEvent()
    reader_ready_event = MPEvent()

    # 3️⃣ 创建推理线程
    print("📦 第3步：创建推理线程...")
    tier1_thread = Tier1_YOLO_Thread()
    tier2_thread = Tier2_YOLO_Thread()
    result_thread = ResultProcessor()

    # 4️⃣ 启动推理线程（开始加载模型）
    print("⏳ 第4步：启动推理线程，等待模型预热...")
    tier1_thread.start()
    tier2_thread.start()

    # 等待模型就绪
    tier1_thread.ready_event.wait(timeout=30)
    tier2_thread.ready_event.wait(timeout=30)
    print("✅ 推理模型已就绪！\n")

    # 5️⃣ 启动VideoReader进程
    print("🚀 第5步：启动VideoReader进程...")
    print(f"   视频路径: {Config.VIDEO_PATH}")
    reader_process = mp.Process(
        target=video_reader_process,
        args=(Config.VIDEO_PATH, frame_index_queue, stop_event_mp, reader_ready_event),
        name="VideoReaderProcess"
    )
    reader_process.start()
    print(f"   进程已启动，PID: {reader_process.pid}")
    print(f"   进程存活: {reader_process.is_alive()}")

    # 等待VideoReader就绪
    print("   等待VideoReader就绪...")
    if not reader_ready_event.wait(timeout=10):
        print("⚠️  警告: VideoReader进程未能在10秒内就绪")
        print(f"   进程存活状态: {reader_process.is_alive()}")
        if not reader_process.is_alive():
            print("❌ VideoReader进程已退出，检查错误")
            print("   请检查视频路径是否正确")
    else:
        print("✅ VideoReader进程已就绪！")

    # 6️⃣ 启动帧分发器（替代Tier0）
    print("🚀 第6步：启动帧分发器...")
    dispatcher = FrameDispatcher(shm_buffer, frame_index_queue, stop_event_mp)
    dispatcher.start()

    # 7️⃣ 启动结果处理器
    print("🚀 第7步：启动结果处理器...\n")
    result_thread.start()

    # 8️⃣ 等待进程完成
    try:
        print("✅ 所有组件已启动，开始处理...\n")
        reader_process.join()  # 等待VideoReader进程结束
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 9️⃣ 清理资源
    print("\n等待所有线程结束...")
    stop_event.set()
    stop_event_mp.set()

    # 先等待dispatcher线程结束（它持有shm_buffer引用）
    dispatcher.join(timeout=5)

    # 等待推理线程结束
    tier1_thread.join(timeout=5)
    tier2_thread.join(timeout=5)
    result_thread.join(timeout=5)

    # 确保子进程完全退出
    if reader_process.is_alive():
        reader_process.terminate()
    reader_process.join(timeout=5)

    # 关闭进程间队列
    try:
        frame_index_queue.close()
        frame_index_queue.join_thread()
    except Exception:
        pass

    # 关闭共享内存（确保所有进程/线程都已退出后再清理）
    print("🧹 清理共享内存...")
    try:
        # 给系统一点时间释放所有引用
        time.sleep(0.5)
        shm_buffer.close()
        shm_buffer.unlink()
        print("✅ 共享内存清理完成")
    except Exception as e:
        print(f"⚠️  共享内存清理警告: {e}")


    print("\n✅ 系统关闭完成（多进程模式）")


def main_singleprocess():
    """单进程模式主函数（原始架构）"""
    print("="*70)
    print("🚀 单进程架构启动")
    print("="*70)
    
    # 🆕 训练模式初始化
    global global_train_collector
    if Config.TRAIN_MODE:
        print("\n⭐ 训练模式已启用")
        if TRAIN_COLLECTOR_AVAILABLE and MOT_EVALUATOR_AVAILABLE:
            try:
                # 先创建 MOT 评估器获取 GT 数据
                eval_config = MOTEvalConfig(
                    iou_threshold=Config.MOT_IOU_THRESHOLD,
                    target_class=Config.MOT_TARGET_CLASS,
                    original_width=640,
                    original_height=640,
                    target_size=640
                )
                temp_evaluator = MOTEvaluator(Config.GT_FILE, eval_config)
                
                # 创建训练数据收集器
                global_train_collector = TrainDataCollector(
                    gt_by_frame=temp_evaluator.gt_by_frame,
                    iou_threshold=Config.MOT_IOU_THRESHOLD
                )
                print("[Main] ✅ 训练数据收集器初始化成功")
            except Exception as e:
                print(f"[Main] ⚠️ 训练数据收集器初始化失败: {e}")
                import traceback
                traceback.print_exc()
                global_train_collector = None
        else:
            print("[Main] ⚠️ 训练模式需要 TrainDataCollector 和 MOTEvaluator 模块")
            global_train_collector = None

    # 1.创建所有线程
    tier0_thread = Tier0_LSTM_Thread()
    tier1_thread = Tier1_YOLO_Thread()
    tier2_thread = Tier2_YOLO_Thread()
    result_thread = ResultProcessor()

    # 2.先启动推理线程（开始加载模型和Warmup）
    print("\n📦 第1步：启动推理模型线程...")
    tier1_thread.start()
    tier2_thread.start()

    # 3. 等待模型加载和Warmup完成
    print("⏳ 第2步：等待模型预热完成...")
    tier1_thread.ready_event.wait(timeout=30)  # 最多等30秒
    tier2_thread.ready_event.wait(timeout=30)
    print("✅ 所有模型已就绪！\n")

    # 4. 启动其他线程
    print("🚀 第3步：启动处理线程...")
    result_thread.start()
    tier0_thread.start()

    # 5. 最后启动视频读取（此时模型已经预热好了）
    print("📹 第4步：启动视频读取...\n")
    try:
        video_reader = VideoReader(Config.VIDEO_PATH)
        video_reader.start()
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 发送停止信号
    stop_event.set()

    # 等待所有线程结束
    print("\n等待所有线程结束...")
    tier0_thread.join(timeout=5)
    tier1_thread.join(timeout=5)
    tier2_thread.join(timeout=5)
    result_thread.join(timeout=5)

    print("\n✅ 系统关闭完成（单进程模式）")



if __name__ == "__main__":
    import argparse
    import os
    import sys

    # 设置 argparse
    parser = argparse.ArgumentParser(description='LSTM Train Mode / Inference System')
    parser.add_argument('dataset_root', nargs='?', help='Path to the MOT dataset root directory (e.g., .../MOT17-02-DPM)')
    args = parser.parse_args()

    # 处理 CLI 参数
    if args.dataset_root:
        dataset_root = args.dataset_root.rstrip(os.sep)  # 去除末尾斜杠
        print(f"[CLI] 收到数据集路径: {dataset_root}")
        
        # 1. 设置 GT 路径 (v3: 使用各个文件夹下的 gt_yolo.txt)
        gt_path = os.path.join(dataset_root, "gt_yolo.txt")
        if os.path.exists(gt_path):
            Config.GT_FILE = gt_path
            print(f"[CLI] ✅ GT文件已设置: {Config.GT_FILE}")
        else:
            print(f"[CLI] ⚠️ GT文件未找到: {gt_path}")
            if Config.TRAIN_MODE:
                print(f"[CLI] ❌ 错误: 训练模式必须提供有效的GT文件")
                # 即使没找到也不强制退出，可能用户想用默认值? 
                # 但CLI意图通常是覆盖。这里给个警告。

        # 2. 设置 Video 路径
        # 尝试寻找 main/*.mp4 或直接寻找 *.mp4
        video_found = False
        
        # 策略1: 检查 dataset_root/main/SEQ_NAME.mp4 (基于用户提供的样本路径)
        seq_name = os.path.basename(dataset_root)
        
        # 3. 设置输出文件名 (使用序列名)
        if seq_name:
            Config.TRAIN_DATA_OUTPUT = f"{seq_name}-new-yolo.json"
            print(f"[CLI] ✅ 训练数据输出文件名已设置为: {Config.TRAIN_DATA_OUTPUT}")

        candidate_video = os.path.join(dataset_root, "main", f"{seq_name}.mp4")
        if os.path.exists(candidate_video):
            Config.VIDEO_PATH = candidate_video
            video_found = True
            print(f"[CLI] 策略1命中: {candidate_video}")
        
        # 策略2: 如果没找到，搜索目录下所有的 mp4 (优先找 main/ 下的)
        if not video_found:
            for root, dirs, files in os.walk(dataset_root):
                for file in files:
                    if file.endswith(".mp4"):
                        Config.VIDEO_PATH = os.path.join(root, file)
                        video_found = True
                        break
                if video_found:
                    break
        
        if video_found:
            print(f"[CLI] ✅ 视频路径已设置: {Config.VIDEO_PATH}")

        else:
            print(f"[CLI] ⚠️ 未在 {dataset_root}下找到MP4视频，保持默认: {Config.VIDEO_PATH}")

    # 设置multiprocessing启动方法（Linux兼容性）
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # 已经设置过了

    # 简单的模式选择逻辑：如果有 train_mode 标志
    if Config.TRAIN_MODE:
        print("[System] ⚠️ 训练模式已启用，强制使用单进程模式以确保数据收集完整性")
        main_singleprocess()
    elif Config.ENABLE_MULTIPROCESSING:
        main()
    else:
        main_singleprocess()


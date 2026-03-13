# 架构

先说下整体，这套代码不是纯多线程，有两条路：

- 正式推理走的是 1 个读视频子进程 + 主进程里几条工作线程，混合的
- 训练数据收集走单进程多线程
- 现在仓库默认 `TRAIN_MODE=True`，你直接跑走的是训练数据收集那条路

## 1. 入口怎么选

入口在 `main.py` 最下面，很简单：

- `TRAIN_MODE=True`：走 `main_singleprocess()`
- `TRAIN_MODE=False` + `ENABLE_MULTIPROCESSING=True`：走 `main_multiprocess()`
- 其他情况：还是 `main_singleprocess()`

就两件事你分清就行：

- 多进程架构代码已经写好了
- 但默认配置跑的是训练模式，不是多进程推理

## 2. 推理模式：Jetson 上怎么跑

真正在 Jetson 上跑推理的时候用的是：

- `TRAIN_MODE=False`
- `ENABLE_MULTIPROCESSING=True`

就是 `main_multiprocess()`

写成这样主要是为了把读视频、前处理和推理调度拆到不同进程/线程上，让 Jetson 多核跑起来，不然全塞一个 Python 解释器里 GIL 会卡住。

分两层：

- `VideoReaderProcess` 是真的子进程，OS 级并发
- 主进程里 `FrameDispatcher / Tier1 / Tier2 / ResultProcessor` 这些是线程，主要负责调度和 GPU 推理

所以就是：

- 读视频和轻量特征提取在独立进程
- GPU 推理和结果处理在主进程
- 两边靠共享内存和轻量 IPC 同步

## 3. 多进程推理这条路

这个模式下不是全多进程，准确说是：

- `VideoReaderProcess` 子进程读视频 + 做轻量通道选择
- 主进程里 `FrameDispatcher`、`Tier1_YOLO_Thread`、`Tier2_YOLO_Thread`、`ResultProcessor` 这些线程

主干多进程，推理还是主进程内的多线程流水线。

### 3.1 结构图

```text
VideoReaderProcess
├── 读视频帧
├── ReaderFeatureExtractor 提轻量特征
├── 非阻塞读取最新 StateSnapshot
├── 组装 ChannelSelectorInput
├── 运行占位版 selector（现在还不是正式 LSTM）
└── 把帧和 metadata 写入 SharedMemoryFrameBuffer
    └── frame_index_queue 通知主进程哪个 slot 可读

Main Process
├── FrameDispatcher
│   ├── 从共享内存读帧和 metadata
│   ├── action=SKIP_PREDICT -> 直接生成 Kalman 结果
│   ├── action=SKIP_GMC -> 直接生成 GMC 结果
│   └── action=INVOKE_TIER1 -> 发给 Tier1
├── Tier1_YOLO_Thread
│   ├── 跑 YOLO11n
│   ├── 置信度够 -> 结果进 result_queue
│   └── 置信度不够 -> 发给 Tier2
├── Tier2_YOLO_Thread
│   ├── 跑 YOLO11m 全图
│   └── 或按 ROI 跑 320 模型
└── ResultProcessor
    ├── 消费 result_queue
    ├── 统计和显示
    ├── 只在有框的 GPU 结果上更新 latest_detection_boxes / Kalman tracker
    └── 生成 StateSnapshot 回推给 reader 进程
```

### 3.2 每层干什么

`VideoReaderProcess` 不是完整的 Tier0，它就干三件事：

- 读帧
- 算轻量帧级特征
- 做通道选择占位逻辑，把结果交给主进程

真正的 CPU 结果生成在主进程 `FrameDispatcher` 里：

- `SKIP_PREDICT` 走 Kalman prediction
- `SKIP_GMC` 走 GMC prediction

就是说现在多进程路径里 selector 在 reader 进程，Kalman / GMC 执行在主进程，这个边界你知道一下就行，别把 LSTM 和 tracker 混一起。

### 3.3 进程间通信

多进程这边有这几条 IPC：

| 通道 | 类型 | 方向 | 干嘛的 |
|------|------|------|------|
| `SharedMemoryFrameBuffer` | 共享内存 ring buffer | reader -> main | 帧数据和 metadata 都在这 |
| `frame_index_queue` | `MPQueue` | reader -> main | 传 slot 索引，轻量通知 |
| `state_snapshot_queue` | `MPQueue(maxsize=1)` | main -> reader | 只保留最新的 `StateSnapshot` |
| `stop_event_mp` | `MPEvent` | main -> reader | 停止信号 |
| `reader_ready_event` | `MPEvent` | reader -> main | reader 就绪信号 |

`SharedMemoryFrameBuffer` 里写的 metadata：

- `frame_id`
- `timestamp`
- `lstm_score`
- `motion_dx`
- `motion_dy`
- `action`

reader 侧算出来的全局运动也一起带过去了，不是只传一个 score。

`lstm_score` 这个字段是占位实现顺手带的，后面不一定要保留 score -> action 这套，直接做三分类 action 也行。

### 3.4 共享内存怎么同步

简单说就是：

- 帧先写进共享内存 slot
- 然后把 `slot_idx` 放进 `frame_index_queue`
- 主进程拿到 `slot_idx` 再去共享内存读

所以这个 queue 不是传帧的，是做"这个 slot 写完了你可以读了"的轻量通知。

现在还没做 consumer 读完后回收确认给 producer 的 ack 队列，ring buffer 固定循环覆盖，配合 queue 顺序消费来用。

### 3.5 `StateSnapshot` 干嘛的

reader 进程拿不到主进程里的 tracker 状态，所以补了条反向同步：

- `ResultProcessor` 处理完 GPU 结果后构建 `StateSnapshot`
- 通过 `state_snapshot_queue` 推给 reader
- reader 每帧用 `drain_latest_state_snapshot()` 非阻塞取最新的

里面主要是：

- 最近 GPU 修正信息
- tracker 当前状态摘要

后面 LSTM 直接拿这个结构化摘要就够了，不用碰主进程对象。

## 4. 单进程训练模式

这条路是你现在默认配置跑的那个。

没有 reader 子进程，全在一个进程里，靠线程和 `queue.Queue` 串。

### 4.1 结构图

```text
Main Thread
└── VideoReader.start()
    └── frame_queue

Tier0_LSTM_Thread
├── 从 frame_queue 取帧
├── 占位版 _lstm_predict_score()
├── _estimate_motion_vector()
└── 如果 TRAIN_MODE=True
    └── _run_train_mode()
        ├── 先跑 Kalman 结果
        ├── 再跑 GMC 结果
        ├── 再把同一帧送去 Tier1/Tier2 跑推理
        └── 从 train_result_queue 等真实推理结果回来

Tier1_YOLO_Thread / Tier2_YOLO_Thread
└── 推理结果同时写 result_queue
    └── 训练模式下也会额外写 train_result_queue

ResultProcessor
├── 处理 result_queue
├── 更新 tracker / latest detections
└── 结束时触发 TrainDataCollector 保存 JSON
```

### 4.2 训练数据怎么收的

训练模式下 `Tier0_LSTM_Thread._run_train_mode()` 每帧做三件事：

1. 记下这帧的 `lstm_score` 和 `motion_vec`
2. 本地生成 `kalman` 和 `gmc` 两路结果
3. 同一帧送 Tier1/Tier2，等 `train_result_queue` 回来真正推理结果

`TrainDataCollector` 把三路结果和 GT 对比，给出每路 metrics、`best_channel`、导出训练 JSON。

所以训练模式不是在线训练，是离线采样打标签。

## 5. LSTM 在哪

这个容易搞混，说清楚。

LSTM 就是个轻量的时序通道选择器，决定当前帧走 `SKIP_PREDICT` / `SKIP_GMC` / `INVOKE_TIER1`，不是 detector 也不是 tracker。

现在：

- 多进程路径：selector 在 `video_reader_process`
- 单进程路径：selector 占位在 `Tier0_LSTM_Thread`
- Kalman / GMC 生成框是后面 CPU prediction 的事，不是 LSTM 的

你要改的就是 selector 这层。

## 6. 已经接好的 LSTM 输入接口

这部分代码里都有了，不是口头约定。

### 6.1 reader 本地特征

`channel_selector_interfaces.py` 里 `ReaderFeatureExtractor` 每帧产出：

- `frame_diff_mean`
- `frame_diff_std`
- `global_motion_dx`
- `global_motion_dy`
- `optical_flow_valid_ratio`
- `optical_flow_residual`
- `tracked_point_count`
- `is_bootstrap_frame`

### 6.2 主进程同步过来的

`ResultProcessor` 构建 `StateSnapshot` 推过来的：

- 最近一次 GPU 修正信息
- 最近几次 GPU 检测框数变化
- tracker 数量和确认状态
- track age / time since update
- speed / uncertainty
- prediction error 统计

### 6.3 统一输入

`ChannelSelectorInput` 把上面两个合一起了，直接吃这个就行，不用自己拆数据。

## 7. 哪些地方还是占位

说清楚免得你以为都做完了。

现在 placeholder 主要两块：

- 多进程路径里 `video_reader_process` 直接拿 `frame_diff_mean` 当 `lstm_score`
- 单进程路径里 `Tier0_LSTM_Thread._lstm_predict_score()` 也是占位

现在那套 `score + threshold` 只是让流程能跑的兼容写法，你可以继续输出 score 自己映射，也可以直接输出 action。

已经完成的：

- 架构
- 数据通路
- IPC
- 训练数据收集
- LSTM 输入接口

还没做的：

- 正式的时序模型
- 正式的在线通道选择策略

## 8. 评估和训练数据对齐

现在默认按 `640x640` 视频跑，样例视频也是这个尺寸。

评估传给 `MOTEvaluator` 的配置：

- `original_width=640`
- `original_height=640`
- `target_size=640`

就是 GT 已经和 640 视频对齐的用法，不是按原始 MOT 的 `1920x1080` 再做 letterbox。后面如果要用原始 MOT `gt.txt` 对齐原始图片尺寸，评估配置要单独改。

## 9. 文件清单

| 文件 | 干嘛的 |
|------|------|
| `main.py` | 主程序，入口、两条运行路径、调度逻辑 |
| `channel_selector_interfaces.py` | LSTM 输入接口和 reader 侧特征提取 |
| `kalman_tracker.py` | 多目标 Kalman tracker |
| `train_collector.py` | 训练模式下三路结果对比 GT 导出 JSON |
| `mot_evaluator.py` | GT 解析和指标评估 |
| `data/` | 跑好的三组训练 JSON |

## 10. 容易搞混的点

- "多进程"说的是推理路径主干，系统里还是有线程的
- 默认配置跑的是单进程训练模式，不是多进程推理
- 部署时想释放 Jetson 多核、绕 GIL，看 `main_multiprocess()` 那条路
- reader 进程负责 selector 输入和占位决策，不管 tracker 更新
- tracker 更新在 `ResultProcessor`
- `StateSnapshot` 是主进程推给 reader 的，不是 reader 自己算的
- LSTM 改的是 selector，不要把 Kalman / YOLO / ResultProcessor 一起动

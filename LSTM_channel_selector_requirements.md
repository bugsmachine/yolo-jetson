# LSTM 通道选择器

这个文档说一下 LSTM 这部分要做什么，接口我基本都写好了，你直接拿来用就行。你只要做和神经网络这部分的东西就可以了。

## 1. 你要做的是什么

简单说就是一个 **channel selector**，不是做检测，是做调度。

根据最近一段时间的时序信息，判断当前帧该走哪条路：

- `SKIP_PREDICT` — 只走 CPU Kalman 预测，不调 GPU
- `SKIP_GMC` — 走全局运动补偿的 CPU 预测
- `INVOKE_TIER1` — 触发一次 GPU 检测

Tier2 不用管，那个是 Tier1 置信度不够的时候自动升级的，LSTM 不需要管它。

核心就是：**这帧值不值得做 GPU 校正，还是继续信 CPU 预测。**

## 2. 期望的行为

### 2.1 稳定场景要省电

直道、目标运动规律、相机稳定的时候：

- 大部分帧只走 CPU 预测
- 每隔 1~2 秒做一次 GPU 修正就行
- GPU 是"校正器"，CPU 预测是"主路径"

### 2.2 相机运动大的时候要谨慎

手持拍摄、相机晃动这种场景：

- 自动降低对 CPU 预测的信任
- 更频繁触发 GPU 修正

注意一个坑：不能只看帧间相似度高就跳过 GPU，因为相机持续运动的时候帧间变化可能不大，但目标坐标误差会一直积累。

所以 LSTM 要有自适应能力：最近稳定就敢跳，最近运动大/风险高就缩短 GPU 修正间隔。

## 3. 模型设计

### 3.1 放在 reader 进程

LSTM 放在 `video_reader_process` 里（在 `main.py` 里搜这个函数名就行）。

流程是：
1. reader 进程读帧
2. 提取轻量特征
3. LSTM 输出通道决策
4. 帧 + 决策写入共享内存

### 3.2 不要吃原始图像

不要让 LSTM 直接读完整图像。输入应该是结构化的低维特征（后面会说具体有哪些，接口都写好了）。

LSTM 负责时序决策，detector 负责识别，tracker 负责状态传播，不要混成一个大模型。

### 3.3 时序记忆 1~2 秒

LSTM 不能只看当前一帧，需要保持最近 1~2 秒的记忆。按 30 FPS 算大概 30~60 帧。

实现方式不限，LSTM hidden state、GRU、滑动窗口都行，重点是根据最近几十帧的趋势来判断，不是只看当前帧。

## 4. 输入特征

我已经把特征提取和接口都写好了
### 4.1 帧级特征 — 直接在 reader 里算

这部分代码在 `channel_selector_interfaces.py` 的 `ReaderFeatureExtractor` 类里，已经实现了

`video_reader_process` 每帧会调用 `feature_extractor.extract(frame, frame_id)`，返回一个 `FrameFeatureSnapshot`，包含：

| 字段 | 含义 |
|------|------|
| `frame_diff_mean` | 灰度帧差均值（归一化到 0~1） |
| `frame_diff_std` | 灰度帧差标准差 |
| `global_motion_dx` | 全局运动向量 X（光流中值） |
| `global_motion_dy` | 全局运动向量 Y |
| `optical_flow_valid_ratio` | 光流跟踪成功率 |
| `optical_flow_residual` | 光流残差（运动一致性指标） |
| `tracked_point_count` | 成功跟踪的特征点数 |
| `is_bootstrap_frame` | 是否为首帧 |

用 `frame_features.to_dict()` 可以直接拿到 dict 格式。

### 4.2 Tracker / 检测摘要 — 主进程同步过来的

这部分 reader 进程自己算不了，是主进程那边的 `ResultProcessor` 每次处理完 GPU 检测结果后，通过 `state_snapshot_queue`（一个 `MPQueue`）推过来的

数据结构是 `StateSnapshot`，定义也在 `channel_selector_interfaces.py` 里，字段：

| 字段 | 含义 |
|------|------|
| `last_gpu_frame_id` | 上一次 GPU 检测是哪一帧 |
| `last_gpu_source` | 上一次 GPU 检测的来源（tier1/tier2） |
| `last_gpu_box_count` | 上一次 GPU 检测出多少框 |
| `last_gpu_max_conf` | 上一次 GPU 检测最大置信度 |
| `recent_gpu_box_count_mean` | 最近几次 GPU 检测框数均值 |
| `recent_gpu_box_count_delta` | 最近两次 GPU 检测框数变化 |
| `tracker_count` | 当前跟踪器数量 |
| `confirmed_tracker_count` | 确认的跟踪器数量 |
| `mean_track_age` | 平均 track 存活时间 |
| `mean_time_since_update` | 平均多久没被 GPU 更新 |
| `max_time_since_update` | 最久没被更新的 track |
| `mean_speed` | 平均目标速度 |
| `mean_position_uncertainty` | 平均位置不确定度（Kalman 协方差） |
| `max_position_uncertainty` | 最大位置不确定度 |
| `prediction_error_ma` | 预测误差移动平均 |
| `prediction_error_p95` | 预测误差 P95 |

另外 `StateSnapshot` 还有个方法 `frames_since_last_gpu(current_frame_id)` 可以算距离上次 GPU 修正过了多少帧。

用 `state_snapshot.to_dict(current_frame_id)` 可以拿到 dict 格式（包含 `frames_since_last_gpu`）。

### 4.3 组装好的统一输入

`ChannelSelectorInput` 把上面两个合在一起了，也在 `channel_selector_interfaces.py` 里：

```python
selector_input = ChannelSelectorInput(
    frame_features=frame_features,
    state_snapshot=latest_state_snapshot,
)

# 拿到全部特征的 dict
features = selector_input.to_feature_dict()

# 拿到特征名列表
names = selector_input.feature_names()
```

这个 `to_feature_dict()` 返回的 dict 包含上面两个表所有字段，你的 LSTM 直接用这个 dict 的 values 作为输入向量就行。

### 4.4 reader 进程里的调用流程

`video_reader_process` 里每帧已经在做这些了（你可以搜 `main.py` 里 `video_reader_process` 函数看）：

```python
# 1. 提取帧级特征
frame_features = feature_extractor.extract(frame, frame_id)

# 2. 非阻塞读取最新的 tracker 状态
latest_state_snapshot = drain_latest_state_snapshot(
    state_snapshot_queue, latest_state_snapshot
)

# 3. 组装输入
selector_input = ChannelSelectorInput(
    frame_features=frame_features,
    state_snapshot=latest_state_snapshot,
)
```

现在第 4 步是个 **placeholder**，没有真正的 LSTM，纯假的。你看 `main.py` 里 `video_reader_process` 现在只是先写成下面这样：

```python
lstm_score = selector_input.frame_features.frame_diff_mean
```

现在是直接拿帧差均值当 score，然后再用固定阈值映射 action：

```python
if lstm_score < Config.LSTM_THRESHOLD_LOW:      # < 0.3
    action_type = ActionType.SKIP_PREDICT.value
elif lstm_score < Config.LSTM_THRESHOLD_HIGH:    # < 0.7
    action_type = ActionType.SKIP_GMC.value
else:
    action_type = ActionType.INVOKE_TIER1.value
```

这个完全没有时序能力，就是个占位让流程能跑起来的。你要做的就是把这块替换掉。

这里不强制你一定还要保留 `score -> 阈值 -> action` 这套形式：

- 你可以继续输出一个连续分数，再自己定义映射方式
- 也可以直接输出三分类 action

只要最后能稳定决定走哪条通道就行。

单进程路径里也有一个类似的占位，在 `Tier0_LSTM_Thread._lstm_predict_score()` 里（大概 386 行），也是纯帧差，同样需要替换。

现在代码里那两个阈值只是为了先把接口和流程跑通，不是要你必须按这个形式做。

### 4.5 调度历史特征

这部分接口里没有，你自己在 LSTM 模块里维护就行：

- 最近 N 帧中触发 GPU 的次数
- 连续跳过 GPU 的帧数
- 最近几次 GPU 修正是否明显改善了预测误差

这些用 `StateSnapshot` 里的字段自己统计就可以，不需要额外接口。

## 5. 训练数据

训练数据收集我也写好了，在 `train_collector.py` 的 `TrainDataCollector` 里。

开启 `Config.TRAIN_MODE = True` 后，系统每帧会同时跑三个通道（Kalman / GMC / Inference），对比 GT 算 F1，然后标出每帧的最佳通道，最后导出成 JSON。

你可以用这个 JSON 来训练，label 就是 `best_channel` 字段。但是你这边没法自己做我之前给ct了三个json数据在data文件夹下你先验证下可行性然后找适合我们的数据跑yologt这部分在readme里说了就是一个相对的正确答案 然后我再拿着这些文件夹在机器上把真实的数据跑出来 因为你的工作只有这一部分是要和机器接触的 我公寓的网不好打洞只能我来跑。

## 6. 决策原则

### 低风险 → 省电

目标数稳定、运动方向稳定、相机运动小、预测误差低 → 走 `SKIP_PREDICT` 或 `SKIP_GMC`

### 风险升高 → 修正

相机持续运动、新目标进入、遮挡增加、轨迹不稳定、tracker 不确定度升高、连续跳过太久 → 走 `INVOKE_TIER1`

### 硬约束

即使 LSTM 说可以跳过，也要保留一些兜底规则：

- 最大连续 skip 帧数限制
- 最大无 GPU 校正时间限制
- 新目标进入边缘区域时强制 GPU
- tracker 不确定度超阈值时强制 GPU

LSTM 是主决策器，但系统要有 guard rails。

## 7. 训练目标

不要只做成“只看单帧”的东西，最好让模型隐式考虑“短期内继续 skip 是否安全”。标签和短期时序风险相关比较好。

可以分两步走：

1. **先做基础版 selector**：输入带时序特征，输出可以是分数，也可以直接是 action
2. **后面再优化**：做“通道选择 + 下次 GPU 修正间隔”的联合决策

## 8. 交付

最后需要的东西：

- LSTM/GRU 模块（能接入 `video_reader_process` 里替换占位逻辑）
- 输入特征定义（直接用 `ChannelSelectorInput` 就行）
- 输出 action 定义（三选一：SKIP_PREDICT / SKIP_GMC / INVOKE_TIER1）
- 基础配置（历史窗口长度、最大 skip 时间、强制 GPU 周期）
- 一版离线评估结果：
  - 稳定场景下 GPU 调用频率有没有降
  - 运动场景下 GPU 修正有没有更频繁
  - 整体比固定帧差阈值合不合理

展示的话我想的是我有收集数据的模块 到时候把数据收集下来用不同颜色的框子区分cpu和推理 正常情况应该是灰色的（cpu推理）占多如何每秒强制走一下gpu

## 9. 总结

就是一个自适应的轻量 selector：

> 基于最近 1~2 秒时序特征，判断当前该不该继续信 CPU 预测，选择走哪个通道。

接口和数据都给你了，你主要做模型这块就行。有问题随时问。

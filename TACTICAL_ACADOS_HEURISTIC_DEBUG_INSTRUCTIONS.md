# Tactical ACADOS Heuristic Debug Instructions

## Goal

请只修改 `tactical_acados` 相关代码，先不要碰 RL，不要大改架构。  
本次任务只做两件事：

1. 找出为什么当前 heuristic 基本一直显示 `FOLLOW_CENTER`
2. 找出为什么在 yasnorth 5号弯附近（约 `s = 1200 ~ 1300`）会出现局部规划不可行、轨迹尖点、轨迹歪斜

---

## Important Constraint

`optimization_horizon_m` **不能在线自适应修改**，因为它会影响求解器结构。  
因此：

- 本次修改中 **不要做在线 horizon 自适应**
- 统一使用固定值：
  - `optimization_horizon_m = 300.0`
- 如果需要测试其他 horizon，只能在**每次仿真开始前**手动改配置，不要在运行过程中动态改

---

## Overall Principle

先做 **诊断和最小修改**，不要先大改 heuristic 逻辑。  
按下面顺序做：

1. **加调试日志**
2. **定位 FOLLOW_CENTER 的真正原因**
3. **定位 5号弯尖点 / 不可行的来源**
4. **只做必要的小改动**
5. 给出结果总结和下一步建议

---

## Files You May Modify

只允许优先修改这些文件：

- `tactical_acados/sim_tactical.py`
- `tactical_acados/policies/heuristic_policy.py`
- `tactical_acados/safe_wrapper.py`
- `tactical_acados/planner_guidance.py`
- `tactical_acados/acados_planner.py`

不要去改 RL 相关文件，不要改整体工程结构。

---

## Task 1: Add Debugging First

### 1.1 In `sim_tactical.py`, log the full tactical chain

在每个 planning cycle 中，记录并打印以下内容：

- `raw_action.discrete_tactic`
- `sanitized_action.discrete_tactic`
- `safe_discrete_set`
- `follow_module_active`
- `planner_healthy`
- `guidance.terminal_n_target`
- `guidance.speed_scale`
- `guidance.safety_distance`
- `min_corridor_width`
- `used_fallback`

如果 heuristic policy 有内部 phase/state，也记录：

- `policy_phase`
- `target_id`
- `gap_to_target`
- `locked_side`
- `phase_time`

### Requirements
- 不要只 `print`
- 也要写到日志结构中，便于后续分析
- 如果当前没有合适日志结构，就新增一个 `debug_log` 字典或列表

### Purpose
确认：

- heuristic 原始动作到底是不是在输出 `PREPARE_OVERTAKE / OVERTAKE`
- 是否被 `safe_wrapper` 打回了 `FOLLOW_CENTER`
- 一旦进入 `FOLLOW`，是否又被 `follow_module` 进一步强化成跟车

---

## Task 2: Diagnose Why It Always Shows FOLLOW_CENTER

### 2.1 Modify `safe_wrapper.py`
重点检查 `_is_discrete_tactic_feasible()` 和相关 safe set 逻辑。

### Problem Hypothesis
当前很可能是：

1. heuristic 原始动作已经想超车
2. 但 `safe_wrapper.sanitize()` 判定不可行
3. 动作被打回 `FOLLOW_CENTER`

### Required Change
把 `PREPARE_OVERTAKE` 和 `OVERTAKE` 的可行性判定分开：

#### For `PREPARE_OVERTAKE`
更宽松：
- 只要前方有目标车
- 只要该侧存在**基本探头空间**
- 不需要和真超车一样严格
- blocking 判定要弱于 `OVERTAKE`

#### For `OVERTAKE`
保持严格：
- 要有足够通道宽度
- 对手不能明显封堵该侧
- 需要比 `PREPARE` 更高的安全要求

### Suggested Parameters
新增独立阈值，不要只复用一个：

- `prepare_min_corridor = 1.6 ~ 2.0`
- `overtake_min_corridor = 2.5`

如果配置文件里已有 `overtake_min_corridor`，可以只在 `safe_wrapper.py` 中增加一个 prepare 的内部阈值，或者把它补进 config。

### Goal
允许系统先进入 `PREPARE_OVERTAKE` 探头阶段，而不是一上来就被打回 `FOLLOW_CENTER`。

---

### 2.2 Modify `sim_tactical.py`
确保 `follow_module` 只作用于真正的 `FOLLOW` 模式。

### Must Enforce
只有在下面条件下才允许调用：

```python
action.mode == TacticalMode.FOLLOW
```

### Do NOT apply follow module to:
- `PREPARE_OVERTAKE`
- `OVERTAKE`
- `DEFEND`

### Goal
避免 probe / prepare 阶段被“跟车限速逻辑”压成真正的跟车。

---

### 2.3 In `heuristic_policy.py`, expose internal decision info
如果 heuristic policy 目前没有内部调试字段，请补上。

至少提供一个可读取的调试结构，例如：

```python
self.debug_info = {
    "phase": ...,
    "target_id": ...,
    "gap": ...,
    "locked_side": ...,
    "phase_time": ...,
    "reason": ...,
}
```

然后在 `sim_tactical.py` 中将其记录。

### Goal
先确认 heuristic 到底是：
- 压根没想超
- 还是想超，但被 safe wrapper 打回了

---

## Task 3: Diagnose the Sharp / Infeasible Trajectory near Turn 5

### 3.1 Modify `acados_planner.py`
在以下函数中增加调试信息：

- `plan()`
- `_try_plan()`
- `_generate_fallback()`

### Log the following:
- `used_fallback`
- `_consecutive_failures`
- `resample_ok`
- `applied_terminal_bias`
- `terminal_n_target`
- `current_horizon_m`
- planner exception message（如果有）

### Requirement
本次固定使用：

```python
optimization_horizon_m = 300.0
```

不要加在线 horizon 自适应逻辑。

### Goal
确认那条怪轨迹到底是：
- 原始 ACADOS 轨迹
- 后处理后的轨迹
- 还是 fallback 轨迹

---

### 3.2 Modify `planner_guidance.py`
在 `_compute_corridor()` 中增加调试量。

### Required Debug Values
记录：

- `min_corridor_width = min(n_left - n_right)`
- `argmin_corridor_stage`
- 对应的 stage 索引
- 对应 stage 的 `s`
- 如方便，可记录最近对手的相对位置

### Goal
确认 5号弯尖点/不可行是不是因为 corridor carve 得过窄。

---

### 3.3 Keep horizon fixed, but test high-curvature protection in bias logic
由于 horizon 不能在线改，本次不要做 horizon 自适应。  
请改为在高曲率区 **减弱 terminal bias 后处理**。

修改 `acados_planner.py` 中 `_apply_terminal_bias()`：

### Add a `bias_gain`
在以下情况下减弱 `terminal_n_target` 的实际作用：

- `upcoming_max_curvature` 高
- corridor 很窄
- planner 当前不健康
- 当前 action 属于 `PREPARE_OVERTAKE` 或 `OVERTAKE`

### Example idea
```python
bias_gain = 1.0
if high_curvature:
    bias_gain *= 0.3
if narrow_corridor:
    bias_gain *= 0.5
if planner_unhealthy:
    bias_gain = 0.0
```

然后把 terminal bias 的实际注入改成：

```python
n_bias = ramp * guidance.terminal_n_target * 0.3 * bias_gain
```

### Goal
避免在高曲率区因为后处理的 lateral bias 把轨迹拉出尖点。

---

## Task 4: Keep Horizon Fixed at 300 During This Task

### Explicit Requirement
本次所有运行都使用固定：

```python
optimization_horizon_m = 300.0
```

### Do NOT implement
- online adaptive horizon
- curvature-based dynamic horizon
- speed-based dynamic horizon

### You may only
- record current horizon in logs
- keep it fixed at 300
- if later需要比较，只能在仿真启动前手动改配置

---

## Task 5: After Modifications, Report Back Clearly

完成修改后，请不要只说“改好了”。  
请明确回答以下问题：

1. heuristic 原始输出中，`PREPARE_OVERTAKE / OVERTAKE` 的比例是多少？
2. `safe_wrapper` 把多少次动作打回了 `FOLLOW_CENTER`？
3. 5号弯异常帧是否进入了 fallback？
4. 异常帧对应的最窄 corridor 宽度是多少？
5. 减弱高曲率区 terminal bias 后，尖点/不可行是否有所减少？
6. 当前最值得继续调的参数有哪些：
   - `prepare_min_corridor`
   - `overtake_min_corridor`
   - `terminal bias gain`
   - `follow module activation`
   - `probe/commit thresholds`

---

## Final Note

这次任务不是最终优化 heuristic，而是：

- 先弄清楚为什么一直 `FOLLOW_CENTER`
- 先弄清楚 5号弯附近尖点 / 不可行来自哪一层
- 让 `PREPARE_OVERTAKE` 更容易进入
- 避免高曲率区后处理把轨迹拉坏

完成这些后，再进入下一轮 heuristic 增强。

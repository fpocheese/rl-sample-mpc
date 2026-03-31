# Tactical RL + ACADOS Local Planning System

这是一个为A2RL方程式赛车开发的两层架构：**战术决策层 (Tactical RL)** + **局部轨迹规划层 (ACADOS)**。

## 1. 工程定位：是一个独立的规划器吗？
**既是，又不是。**
- 它**不**是从头写的一个全新底层规划器。
- 它是一个**外挂式的智能大脑（Wrapper）**。它直接复用了你原来在 `src/` 目录下的赛道处理（`Track3D`）、车辆物理模型（`GGManager`）以及 ACADOS 后端（`LocalRacinglinePlanner`）。
- **它需要调用外部函数**。你原来底层的物理约束和优化器丝毫没变，它只负责在更高维度（超车、防守、跟车）给原始的规划器发送**“参数指令（Guidance）”**，从而改变局部最优化的结果（例如从赛车线让开、减速跟车等）。

## 2. 文件夹与文件功能详解

### 核心接口与配置
* `config.py`: 核心参数库。包括所有强化学习、前瞻距离、跟车模块的安全间距设置。
* `tactical_action.py`: 定义了强化学习输出的混合动作空间（离散：超车、防守、跟随；连续：激进程度系数）。
* `observation.py`: 负责构建强化学习的状态空间（29维向量），包括本车状态、对手相对状态和赛道曲率。

### 规划与安全层 (AI与底层规划的桥梁)
* `safe_wrapper.py` **(安全护盾)**: 强制过滤AI的非法指令。如果AI让车撞墙或者做出危险的超车，这里会把动作强制降级为安全的动作（比如跟随）。
* `planner_guidance.py` **(核心接口)**: 将战术动作翻译成规划器能懂的数学约束！**决策层传给规划层的数据就是在这里转换的**。
* `acados_planner.py`: 你原始ACADOS规划器的包装器。它实现了**自适应前瞻距离 (Optimization Horizon)**，并在规划失败时调用跟车模块。
* `follow_module.py`: 独立编写的**跟车模块**。结合 OCP 虚拟墙与后处理速度限制，保证绝对的安全车距，且不会自动降速（保护期）。

### 博弈论与多车环境
* `opponent.py`: 对手预测模块，使用基于 Iterated Best Response (IBR) 的模型来预判对手未来3秒的轨迹。
* `game_value.py` & `rl/theory_prior.py`: **博弈论引入的核心位置**。在此计算 Nash Gap（纳什均衡差距）并将其作为强化学习训练的正则化项，强制让AI学得“聪明且符合博弈规律”。
* `p2p.py`: Push-to-Pass一键加速模块。

### 评估与测试模块
* `metrics.py`: 顶会级别的8项博弈轨迹评价指标（超车成功率、安全性、社会福利、纳什均衡差等）。
* `comparison_test.py`: **3路对比测试框架**，屏幕上并排显示三个视角（我们 vs 无脑循线 vs 传统规则博弈），用于跑论文数据。

### 仿真引擎
* `sim_acados_only.py`: **测试单车 ACADOS 的入口**。这里没有任何AI和决策，纯粹测试底层优化器是否正常。
* `sim_tactical.py`: **测试完整战术AI的入口**。一辆带有AI的主车与多辆对手车进行对抗。
* `scenarios/`: 场景配置文件（A/B局部超车，C为全赛道）。通过修改 `.yml` 轻松切换测试地图和对手位置。

### 训练与强化学习
* `rl/tactical_env.py`: 把它封装成标准 Gymnasium 环境的代码。
* `rl/hybrid_ppo.py`: 我们使用的 Actor-Critic 混合动作PPO算法实现。
* `rl/train.py`: **训练强化学习的入口**。

---

## 3. 使用说明（How to Use）

**环境要求：** `conda activate a2rldet`

### ❓ 如何进行强化学习训练？
运行以下命令即可开始训练PPO网络（自动保存模型到 `checkpoints/`）：
```bash
python tactical_acados/rl/train.py
```

### ❓ 如何单独测试本车的 ACADOS 规划能力（不带AI，不带对手）？
运行以下命令，可视化会在屏幕中显示单车沿 Raceline 规划的过程：
```bash
python tactical_acados/sim_acados_only.py
```
*(如果要关掉动画或者修改步数，直接打开代码拉到最下面 `if __name__ == '__main__':` 修改 `VISUALIZE=False` 等变量即可)*。

### ❓ 如何进行多车对抗的战术测试（直观可视化）？
运行完整的战术仿真器：
```bash
python tactical_acados/sim_tactical.py
```

### ❓ 修改场景怎么切换？
打开 `sim_tactical.py` 或 `comparison_test.py`，滑动到文件最底部的 `if __name__ == '__main__':`：
```python
SCENARIO = 'scenario_a'  # 可以改成 'scenario_b' 或 'scenario_c'
```
你也可以直接去 `tactical_acados/scenarios/` 下面修改 yaml 文件，甚至自己新建对手的起始位置。

### ❓ 提取论文对比数据（Metrics & Comparison）用哪个？
运行 3way 测试台，这会同时在一张图里跑我们算法、无战术算法、普通博弈算法：
```bash
python tactical_acados/comparison_test.py
```
运行结束后，所有8项核心指标会打印到终端，并且在同目录下生成格式化的 `comparison_scenario_a.csv` 文件，方便你画图或者贴论文表格。

---

## 4. 架构解答：决策层到底传给规划层什么？接口在哪？

**接口位置：** `tactical_acados/planner_guidance.py` 中的 `TacticalToPlanner.map()` 函数。

**决策层传递的核心并不是一条完全算好的轨迹，而是“对底层局部优化器的干预与引导参数（Constraints Guidance）”：** 
1. `safety_distance` (安全边界距离): 从默认的膨胀系数放大，比如强行变宽来形成超车走廊。
2. `terminal_n_target` (侧向引导): 如果AI决定“超车_左”，这里会传一个左移的目标值，通过二次函数的偏置（Bias）给ACADOS代价函数施压，骗优化器规划出左侧变道。
3. `speed_scale` / `speed_cap` (速度意图): 与 `follow_module.py` 配合，算出当前能允许的最大上限送进求解器，或者让车逼近理论最高速度前瞻。
4. `optimization_horizon_m` (预设前瞻视距): 现在已固定在稳定的 300 米以防止求解器断层。

这套机制极其优雅且成熟：它**100%保留了你之前论文里建立的所有运动学、打滑避免和最优底盘控制（完美平滑不翻车）**，相当于给原来的局部路径规划器安上了一个懂得看其他对手、选择时机和战术博弈的“上层AI领航员”。

# 赛车线路规划器对比分析：ECOS vs. Acados (SQP)

本报告深入研究了 `local_racing_line_planner` 的实现，并将其与之前在 `traj_planner.cpp` 中使用的基于 ECOS 的方法进行了对比。重点分析了 GGV 约束的实现方式以及求解速度差异的原因。

---

## 1. 对比概览表

| 对比维度 | ECOS 方法 (`traj_planner.cpp`) | Acados 方法 (`local_racing_line_planner.py`) |
| :--- | :--- | :--- |
| **GGV 建模方式** | **分段线性近似 (Piecewise Linear)** | **全非线性约束 (Nonlinear)** |
| **约束平滑度** | 会在“转角”处产生不连续的梯度 | C1 连续的平滑边界 |
| **求解器算法** | 通用内点法 (SOCP) | 专门优化的序列二次规划 (SQP) |
| **热启动 (Warm-start)** | 有限/基本没有 | 高效重用上一帧的状态和乘子 |
| **计算复杂度** | $O(N^3)$ (通用求解器) | **$O(N)$** (利用 Riccati 递归的块结构) |
| **路径与速度耦合** | 解耦进行 (先路径后速度) | **全耦合优化** (同时求解 $s, V, n, \chi$ 等) |
| **工程实现** | 手动构建 CCS 稀疏矩阵 (容易出错) | 通过 CasADi 自动生成高度优化的 C 代码 |

---

## 2. GGV 非线性约束的实现

核心区别在于车辆物理极限（G-G-V 图）的数学表达方式。

### ECOS 方法 (分段线性近似)
在 `traj_planner.cpp` 中，GGV 区域被简化为一组 $(v^2, a_t)$ 空间下的**线性不等式**。

```cpp
// 摘自 traj_planner.cpp: VelOptGlobal
G(LPnum * i + 2, i * nv + 0) = -Cw2_1; // v^2 的系数
G(LPnum * i + 2, i * nv + 1) = -1;      // at 的系数
h(LPnum * i + 2) = Cw0_1;              // 常数
// 结果等效于: -Cw2 * v^2 - at <= Cw0  =>  at >= -Cw0 - Cw2 * v^2
```

*   **逻辑**：通过几条直线（超平面）手动“框出”GGV 的包络。
*   **局限性**：这是一种**保守的近似**。为了保证安全，往往牺牲了一部分车辆性能。此外，线性分段的转角会导致控制量出现跳变。

### Acados 方法 (直接非线性约束)
在 `point_mass_model.py` 中，GGV 被视为真正的**非线性边界**。

```python
# 摘自 point_mass_model.py: diamond 模式
model.con_h_expr = ca.vertcat(
    ax_max - ax_tilde,
    ay_max - ca.fabs(ay_tilde),
    ca.fabs(ax_min) * ca.power(
        (1.0 - ca.power(ca.fabs(ay_tilde) / ay_max, gg_exponent)),
        1.0 / gg_exponent
    ) - ca.fabs(ax_tilde)
)
```

*   **逻辑**：使用幂函数耦合公式（$|a_x/a_{max}|^P + |a_y/a_{max}|^P \le 1$）或直接使用 CasADi 的极坐标插值（Polar Interpolation）。
*   **优势**：在全速度范围内保持**平滑且精确**的包络。求解器可以在不留保守余量的情况下，直接把车辆推向物理极限。

---

## 3. 为何 Acados 求解速度极快且效果更好？

之前的 ECOS 方法效果不佳，主要是受限于其求解器架构和问题构建方式。

### A. 专门的 SQP 算法与热启动
*   **ECOS (SOCP)**：通用求解器，将每次求解视为独立任务，无法有效利用赛车运动的“连续性”。
*   **Acados (SQP)**：利用上一帧的解作为**初始猜测 (Warm-start)**。由于赛车轨迹变化平滑，SQP 通常只需几次（甚至 1 次，即 RTI 模式）迭代即可收敛。

### B. 结构化求解 (Riccati 递归)
*   **ECOS**：你手动构建了巨大的 CCS 稀疏矩阵。在这种方式下，求解器丢失了轨迹优化问题的“时间序列”结构（Markovian property）。
*   **Acados**：使用 **HPIPM** 求解器，其内部使用了 **Riccati 递归**。对于轨迹长为 $N$ 的问题，其计算量与 $N$ 成正比（$O(N)$），而通用稀疏求解器的复杂度通常更高。

### C. 路径与速度的全耦合
*   **旧方法**：通常先优化路径（PathOptLocal）再优化速度（VelOptGlobal）。这种**解耦**会导致路径规划器在挑选线路时，对后续速度规划的物理限制缺乏感知，容易产生无效解。
*   **新方法**：Acados 同时求解所有状态。求解器知道：改变横向偏移（$n$）会改变曲率，进而直接触发非线性 GGV 约束对速度（$V$）的限制。

---

## 结论
Acados 方法之所以大幅优于 ECOS，是因为它在尊重车辆**物理非线性**的同时，采用了与赛车任务完美契合的**数学架构**（SQP + Riccati 递归 + 热启动）。之前的 ECOS 实现则受限于手动构建矩阵的开销以及路径与速度规划之间的脱节。

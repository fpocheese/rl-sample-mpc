import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 文件路径：py 和 csv 在同一目录
# =========================
csv_file_name = "yas_user_smoothed.csv"   # 改成你的文件名

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, csv_file_name)

# =========================
# 2. 读取文件
# =========================
df = pd.read_csv(csv_file)

# =========================
# 3. 读取数据
# =========================
s = df["s_m"].to_numpy()
x = df["x_m"].to_numpy()
y = df["y_m"].to_numpy()
theta = df["theta_rad"].to_numpy()
w_left = df["w_tr_left_m"].to_numpy()
w_right = df["w_tr_right_m"].to_numpy()

# =========================
# 4. 计算法向量
# theta 为中线切向方向角
# 左法向量 = (-sin(theta), cos(theta))
# =========================
nx = -np.sin(theta)
ny =  np.cos(theta)

# =========================
# 5. 计算左右边界
# 注意：
# 你的文件里 w_tr_right_m 是负值
# 所以这里统一都写成：中心线 + 宽度 * 左法向量
# 左边界会到左边，右边界会因为宽度是负值自动到右边
# =========================
x_left = x + w_left * nx
y_left = y + w_left * ny

x_right = x + w_right * nx
y_right = y + w_right * ny

# =========================
# 6. 绘图参数
# =========================
label_interval_s = 50.0
figsize = (10, 10)
fontsize = 8

# =========================
# 7. 绘图
# =========================
plt.figure(figsize=figsize)

plt.plot(x_left, y_left, linewidth=2, label="Left Boundary")
plt.plot(x_right, y_right, linewidth=2, label="Right Boundary")
plt.plot(x, y, "--", linewidth=1.5, label="Centerline")

# =========================
# 8. 每隔一定 s 标注一次
# =========================
s_targets = np.arange(0, s[-1] + label_interval_s, label_interval_s)
used_idx = set()

for s_target in s_targets:
    idx = np.argmin(np.abs(s - s_target))
    if idx in used_idx:
        continue
    used_idx.add(idx)

    plt.scatter(x[idx], y[idx], s=20)
    plt.text(
        x[idx] + 1.0,
        y[idx] + 1.0,
        f"{s[idx]:.0f} m",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
    )

# =========================
# 9. 美化
# =========================
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Track Map with Centerline and Boundaries")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# step3_describe_plot.py
# 目的：描述性统计与可视化，观察糖尿病与非糖尿病组的差异
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取本地数据
print("正在加载数据...")
df = pd.read_csv("data.csv")

# 删除含有缺失值的行
df = df.dropna()

# 删掉无意义的 ID 列
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# 2. 计算均值和中位数
print("\n【按糖尿病分组，计算均值和中位数】")
# 注意：groupby 后 agg 可以同时算多个统计量
stats = df.groupby("Diabetes_binary")[["BMI", "Age", "HighBP", "HighChol"]].agg(["mean", "median"])
print(stats.round(3))

# 3. 画图
# 设置中文字体（Mac 系统通常用 Arial Unicode MS 或 PingFang SC）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 创建画布，1行2列，宽14高6
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1：BMI 按 Diabetes 分组的箱线图
df.boxplot(column="BMI", by="Diabetes_binary", ax=axes[0], grid=False)
axes[0].set_title("BMI 与糖尿病关系 (箱线图)")
axes[0].set_xlabel("Diabetes (0=无, 1=有)")
axes[0].set_ylabel("BMI")
# 去除 boxplot 默认的难看的 title
axes[0].set_title("BMI 与糖尿病关系 (箱线图)")

# 图2：HighBP 按 Diabetes 分组的比例柱状图
# 计算每一组中 HighBP=1 的比例
bp_ratio = df.groupby("Diabetes_binary")["HighBP"].mean()
bp_ratio.plot(kind="bar", ax=axes[1], color=['skyblue', 'salmon'], rot=0)
axes[1].set_title("高血压比例与糖尿病关系 (柱状图)")
axes[1].set_xlabel("Diabetes (0=无, 1=有)")
axes[1].set_ylabel("高血压比例 (HighBP=1)")
axes[1].set_ylim(0, 1) # y轴范围 0 到 1

# 调整布局并显示
plt.suptitle("") # 去掉总标题，因为子图标题已经很清晰
plt.tight_layout()
plt.show()
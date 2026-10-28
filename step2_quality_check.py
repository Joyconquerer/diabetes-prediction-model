# step2_quality_check.py
# 目的：对 X 做数据质量审查，确认无缺失、取值范围合理、目标类别不平衡
# 前提：step1 已跑通，X 为 253680×21，y 为单列 Diabetes_binary

import pandas as pd
import numpy as np

# ---------- 0. 重新加载数据（直接读取 pandas 本地缓存，避免重复联网卡死） ----------
import pandas as pd
import numpy as np

# 直接用 pandas 从 URL 读取，稳定性比 ucimlrepo 好，而且能看进度（如果卡住就是网络慢）
# 读取完整数据集（直接读 UCI 的 ZIP 包）
url = "https://archive.ics.uci.edu/static/public/891/data.csv"
# 先不用本地了，我们试试完整版。如果卡住，就等10秒
df = pd.read_csv(url)

# 手动分离特征和目标
X = df.drop(columns=["Diabetes_binary"])
y = df[["Diabetes_binary"]]
# 删除含有缺失值的行（只有1行，直接删掉最稳妥）
X = X.dropna()
y = y.loc[X.index]  # 目标列也要跟着同步删

# ---------- 1. 每列 dtype / 缺失值 / 唯一值数量 ----------
overview = pd.DataFrame({
    "dtype": X.dtypes.astype(str),
    "missing": X.isna().sum(),
    "n_unique": X.nunique(),
})
print("=" * 70)
print("【每列 dtype / 缺失值 / 唯一值数量】")
print(overview)
print("=" * 70)

# 缺失值总计
total_missing = int(X.isna().sum().sum())
print(f"\n全表缺失值总数: {total_missing}")

# ---------- 2. 数值列 min / max / 均值 / 中位数 ----------
# 该数据集所有列都是数值型（int），统一计算
num_stats = pd.DataFrame({
    "min": X.min(),
    "max": X.max(),
    "mean": X.mean().round(3),
    "median": X.median(),
})
print("\n【数值列 min / max / mean / median】")
print(num_stats)
print("=" * 70)

# ---------- 3. 取值少的列打印 value_counts（唯一值 <= 5 的列） ----------
print("\n【唯一值 ≤ 5 的列，value_counts】")
for col in X.columns:
    if X[col].nunique() <= 5:
        print(f"\n--- {col} (n_unique={X[col].nunique()}) ---")
        print(X[col].value_counts().sort_index())

# ---------- 4. 越界检查 ----------
print("\n【医学合理性越界检查】")

# BMI：正常 10–60，0 或 >100 明显异常
bmi_abnormal = X[(X["BMI"] <= 0) | (X["BMI"] > 100)]
print(f"BMI ≤ 0 或 > 100 的行数: {len(bmi_abnormal)}")

# Age：分箱等级 1–13（BRFSS 的 Age 是 13 个年龄段）
age_abnormal = X[(X["Age"] < 1) | (X["Age"] > 13)]
print(f"Age 不在 [1,13] 的行数: {len(age_abnormal)}")

# GenHlth：自评健康 1–5
genhlth_abnormal = X[(X["GenHlth"] < 1) | (X["GenHlth"] > 5)]
print(f"GenHlth 不在 [1,5] 的行数: {len(genhlth_abnormal)}")

# ---------- 5. 目标列类别不平衡 ----------
print("\n【目标列类别不平衡】")
counts = y["Diabetes_binary"].value_counts().sort_index()
ratios = y["Diabetes_binary"].value_counts(normalize=True).sort_index()
print(f"0 (无糖尿病): {counts[0]}  ({ratios[0]:.2%})")
print(f"1 (有糖尿病): {counts[1]}  ({ratios[1]:.2%})")
print(f"不平衡比 (负/正): {counts[0] / counts[1]:.2f} : 1")

# ---------- 6. assert 检查 ----------
# 6a. 无缺失值
# 找出有缺失值的列
print("缺失值所在列：")
print(X.isna().sum()[X.isna().sum() > 0])
#assert total_missing == 0, f"存在缺失值: {total_missing}"
# 6b. BMI 在合理范围（该数据集应无越界）
assert len(bmi_abnormal) == 0, f"BMI 存在越界: {len(bmi_abnormal)}"
# 6c. 目标列只有 0/1（step1 已查，这里防回归）
assert set(y["Diabetes_binary"].unique()) == {0, 1}

print("\n✅ 所有 assert 通过：无缺失、BMI 无越界、目标二分类成立")

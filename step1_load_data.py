# step1_load_data.py
# 目的：通过 ucimlrepo 拉取 UCI 数据集 id=891（CDC Diabetes Health Indicators）
# 验证：特征表形状、目标列取值分布、列名是否包含预期医学变量

import pandas as pd
from ucimlrepo import fetch_ucirepo

# ---------- 1. 拉取数据集 ----------
# fetch_ucirepo 返回一个对象，包含 .data.features 和 .data.targets
cdc = fetch_ucirepo(id=891)

# 取出 features 和 targets（注意：不同版本返回的 DataFrame 结构可能略有差异）
X = cdc.data.features
y = cdc.data.targets

# ---------- 2. 打印形状 ----------
print("=" * 60)
print("特征表形状 (X):", X.shape)
print("目标表形状 (y):", y.shape)
print("=" * 60)

# ---------- 3. 打印目标列取值分布 ----------
# targets 可能是一列或多列，先看列名
print("\n目标表列名:", list(y.columns))
print("\n目标列 Diabetes_binary 的取值分布：")
print(y["Diabetes_binary"].value_counts())
print("\n比例：")
print(y["Diabetes_binary"].value_counts(normalize=True))

# ---------- 4. 打印特征表前 5 行 ----------
print("\n特征表前 5 行：")
print(X.head())

# ---------- 5. 打印特征列名列表 ----------
print("\n特征列名（共 %d 列）：" % len(X.columns))
for i, col in enumerate(X.columns):
    print(f"  {i+1:2d}. {col}")

# ---------- 6. assert 检查 ----------
# 6a. 行数应在 250000–260000 之间（UCI 标注 253680）
assert 250000 <= X.shape[0] <= 260000, \
    f"行数异常：期望 250000–260000，实际 {X.shape[0]}"

# 6b. 目标列取值只有 0 和 1
unique_vals = set(y["Diabetes_binary"].unique())
assert unique_vals == {0, 1}, \
    f"目标列取值异常：期望 {{0, 1}}，实际 {unique_vals}"

print("\n✅ 所有 assert 通过")

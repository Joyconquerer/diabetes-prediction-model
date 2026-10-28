# step4_split.py
# 目的：划分训练集和测试集（80/20），保证正负样本比例一致
import pandas as pd
from sklearn.model_selection import train_test_split

print("正在加载数据...")
df = pd.read_csv("data.csv")

# 清理数据：删除缺失值、删除 ID 列
df = df.dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# 分离特征和目标
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 划分数据集
# test_size=0.2 表示测试集占 20%
# random_state=42 保证每次运行切出来的数据一模一样（固定随机种子）
# stratify=y 极其重要，保证切分后训练集和测试集里，无糖尿病和有糖尿病的比例都维持在 86:14
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 打印形状
print("\n【数据划分结果】")
print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")

# 打印目标均值（即正样本比例）
print(f"\n训练集目标均值（正类比例）: {y_train.mean():.4f}")
print(f"测试集目标均值（正类比例）: {y_test.mean():.4f}")

# assert 检查
assert len(X_train) + len(X_test) == len(X), "划分后总行数不匹配"
assert abs(y_train.mean() - y_test.mean()) < 0.01, "分层抽样后，正负样本比例差异过大"
print("\n✅ 划分成功，比例一致")
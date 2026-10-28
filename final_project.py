import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("1. 正在加载数据并清理...")
df = pd.read_csv("data.csv")
df = df.dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
print(f"数据形状: {X.shape}, 正类比例: {y.mean():.4f}")

print("\n2. 数据统计与画图...")
stats = df.groupby("Diabetes_binary")[["BMI", "Age", "HighBP", "HighChol"]].agg(["mean", "median"])
print(stats.round(3))

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
df.boxplot(column="BMI", by="Diabetes_binary", ax=axes[0], grid=False)
axes[0].set_title("BMI 与糖尿病关系")
bp_ratio = df.groupby("Diabetes_binary")["HighBP"].mean()
bp_ratio.plot(kind="bar", ax=axes[1], color=['skyblue', 'salmon'], rot=0)
axes[1].set_title("高血压比例与糖尿病关系")
plt.suptitle("")
plt.show()

print("\n3. 划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n4. 训练逻辑回归模型...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC 值: {roc_auc_score(y_test, y_prob):.4f}")
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

print("\n5. 特征系数前 10 名:")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coef', ascending=False).drop(columns=['Abs_Coef'])
print(coef_df.head(10).to_string(index=False))
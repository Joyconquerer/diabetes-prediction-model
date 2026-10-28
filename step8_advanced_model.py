import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. 加载数据
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 2. 核心改进 1：使用 class_weight='balanced' 惩罚模型对多数类的偏好
# ---------------------------------------------------------
print("正在训练带权重惩罚的逻辑回归模型...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# 获取预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 3. 核心改进 2：阈值调整（Threshold Moving）
# 默认阈值是 0.5，现在我们把它降到 0.35（让模型更容易报警）
# ---------------------------------------------------------
threshold = 0.35
y_pred_custom = (y_prob >= threshold).astype(int)

# 4. 输出评估结果
print("\n" + "="*50)
print(f"模型 AUC 值: {roc_auc_score(y_test, y_prob):.4f}")
print(f"当前使用的分类阈值: {threshold}")

print("\n【改进后的混淆矩阵】")
cm = confusion_matrix(y_test, y_pred_custom)
print("           预测:无  预测:有")
print(f"实际:无      {cm[0][0]}      {cm[0][1]}")
print(f"实际:有      {cm[1][0]}      {cm[1][1]}  <-- 看左下角的漏诊人数")

print("\n【改进后的分类报告】")
print(classification_report(y_test, y_pred_custom, target_names=["无糖尿病", "有糖尿病"]))
# step5_model.py
# 目的：训练逻辑回归模型，输出评估指标，并查看特征系数
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np

# 1. 加载并清理数据（和上一步完全一样）
print("正在加载数据并划分...")
df = pd.read_csv("data.csv")
df = df.dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 划分（使用 stratify 保证一致性）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 训练逻辑回归模型
print("正在训练逻辑回归模型...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. 在测试集上预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # 取概率值用于计算 AUC

# 4. 输出评估指标
print("\n" + "=" * 50)
print("【模型评估指标】")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC 值 (ROC-AUC): {roc_auc_score(y_test, y_prob):.4f}")

print("\n【混淆矩阵】")
print("格式：")
print("           预测:无  预测:有")
print("实际:无      TN      FP")
print("实际:有      FN      TP")
print(confusion_matrix(y_test, y_pred))

print("\n【分类报告】")
print(classification_report(y_test, y_pred, target_names=["无糖尿病(0)", "有糖尿病(1)"]))

# 5. 查看特征系数
print("\n【特征系数绝对值排名前十】")
# model.coef_[0] 是模型的系数，X.columns 是特征名
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
# 按绝对值排序
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coef', ascending=False).drop(columns=['Abs_Coef'])

# 只打印前 10 行
print(coef_df.head(10).to_string(index=False))
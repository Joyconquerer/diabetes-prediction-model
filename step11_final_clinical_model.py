import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. 加载数据
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("正在训练【临床代价敏感】随机森林模型...")
# 权重调整为 1:3（漏诊代价是误诊的 3 倍，这比较符合现实，不至于走火入魔）
rf_clinical = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    class_weight={0: 1, 1: 3},  
    random_state=42, 
    n_jobs=-1
)
rf_clinical.fit(X_train, y_train)

y_prob = rf_clinical.predict_proba(X_test)[:, 1]

# 2. 科学的阈值寻找方法：Youden's J 统计量
# J = 真正率(Recall) + 真负率(Specificity) - 1
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# 找到让 J 最大的那个阈值索引
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

print("\n" + "="*50)
print(f"临床模型 AUC 值: {roc_auc_score(y_test, y_prob):.4f}")
print(f"基于 Youden's J 算出的最佳科学阈值: {best_threshold:.4f}")

# 3. 预测
y_pred_clinical = (y_prob >= best_threshold).astype(int)

print("\n【最终临床模型混淆矩阵】")
cm = confusion_matrix(y_test, y_pred_clinical)
print("           预测:无  预测:有")
print(f"实际:无      {cm[0][0]}      {cm[0][1]} (误报)")
print(f"实际:有      {cm[1][0]} (漏诊)      {cm[1][1]}")
print("\n【最终分类报告】")
print(classification_report(y_test, y_pred_clinical, target_names=["无糖尿病", "有糖尿病"]))
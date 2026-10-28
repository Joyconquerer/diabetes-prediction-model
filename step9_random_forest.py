import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. 加载数据
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("正在训练随机森林模型 (这可能需要几秒钟)...")
# 2. 训练随机森林
# n_estimators=100 代表种 100 棵树；class_weight='balanced' 同样处理不平衡
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 3. 预测概率
y_prob = rf_model.predict_proba(X_test)[:, 1]

# 4. 阈值调整 (为了对比，我们依然用 0.35)
threshold = 0.35
y_pred_custom = (y_prob >= threshold).astype(int)

# 5. 输出评估结果
print("\n" + "="*50)
print(f"随机森林 AUC 值: {roc_auc_score(y_test, y_prob):.4f}")
print(f"当前使用的分类阈值: {threshold}")

print("\n【随机森林混淆矩阵】")
cm = confusion_matrix(y_test, y_pred_custom)
print("           预测:无  预测:有")
print(f"实际:无      {cm[0][0]}      {cm[0][1]}")
print(f"实际:有      {cm[1][0]}      {cm[1][1]}  <-- 看左下角的漏诊人数")

print("\n【随机森林分类报告】")
print(classification_report(y_test, y_pred_custom, target_names=["无糖尿病", "有糖尿病"]))

# 6. 寻找最重要的特征（树模型的专属功能）
print("\n【随机森林认为最重要的特征 Top 10】")
importances = rf_model.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print(feat_imp.head(10))
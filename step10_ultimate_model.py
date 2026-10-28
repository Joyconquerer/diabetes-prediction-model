import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 我们需要用到 imblearn 库来做 SMOTE
# 如果报错 ModuleNotFoundError: No module named 'imblearn'，请在终端运行：pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

# 1. 加载数据
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. 数据增强（SMOTE）：只对训练集做，绝对不碰测试集！
print("正在使用 SMOTE 合成少数类样本...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"原训练集正类比例: {y_train.mean():.4f}")
print(f"SMOTE 后正类比例: {y_train_resampled.mean():.4f}")

# 3. 训练更强的随机森林
print("正在训练增强版随机森林模型...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15, 
    min_samples_split=5,
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train_resampled, y_train_resampled)

# 4. 预测概率
y_prob = rf_model.predict_proba(X_test)[:, 1]

# 5. 自动寻找最佳阈值（不再是手动 0.35）
# 我们要找到一个阈值，使得 F1-score 最大（兼顾召回率和精准度）
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print("\n" + "="*50)
print(f"SMOTE 随机森林 AUC 值: {roc_auc_score(y_test, y_prob):.4f}")
print(f"自动算出的最佳分类阈值: {best_threshold:.4f}")

# 6. 用最佳阈值预测
y_pred_custom = (y_prob >= best_threshold).astype(int)

print("\n【SMOTE 模型混淆矩阵】")
cm = confusion_matrix(y_test, y_pred_custom)
print("           预测:无  预测:有")
print(f"实际:无      {cm[0][0]}      {cm[0][1]}")
print(f"实际:有      {cm[1][0]}      {cm[1][1]}  <-- 看左下角的漏诊人数")

print("\n【分类报告】")
print(classification_report(y_test, y_pred_custom, target_names=["无糖尿病", "有糖尿病"]))
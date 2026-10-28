import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. 加载数据（沿用你刚才的清理逻辑）
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 2. 划分测试集（必须切分，不能拿训练的题目考试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 训练最终的模型
FINAL_PARAMS = {
    'n_estimators': 141,
    'max_depth': 8,
    'class_weight': {0: 1.0, 1: 5.6187},
    'random_state': 42
}
BEST_THRESHOLD = 0.5920

model = RandomForestClassifier(**FINAL_PARAMS, n_jobs=-1)
model.fit(X_train, y_train)

# ==========================================
# 第一部分：模型是怎么判定的？（透视大脑）
# ==========================================
print("="*50)
print("🧠 模型的判定逻辑：各特征的重要性排名")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print(feat_imp.head(10))

# 画个特征重要性柱状图
plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='barh', color='teal')
plt.gca().invert_yaxis() # 让第一名在最上面
plt.title("模型判定糖尿病时最依赖的 Top 10 特征")
plt.xlabel("重要性分数")
plt.tight_layout()
plt.show()

# ==========================================
# 第二部分：模型考得怎么样？（真实考核）
# ==========================================
print("\n" + "="*50)
print("📝 对 2万+ 名测试集病人的最终考核")

# 拿到预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# 用 0.5920 的阈值进行切割
y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

print(f"在本次测试中，使用的判定阈值为: {BEST_THRESHOLD}")
print(f"整体 AUC 评分: {roc_auc_score(y_test, y_prob):.4f}\n")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("【混淆矩阵】")
print(f"               预测健康     预测患病")
print(f"实际健康(TN): {cm[0][0]:>8} {cm[0][1]:>10} (误报)")
print(f"实际患病(FN): {cm[1][0]:>8} {cm[1][1]:>10} (真阳性)")

# 分类报告
print("\n【分类报告】")
report = classification_report(y_test, y_pred, target_names=["健康(0)", "患病(1)"], digits=4)
print(report)

# 计算具体的漏诊率
miss_rate = cm[1][0] / (cm[1][0] + cm[1][1])
print(f"💡 核心临床指标：在真正的患病者中，模型的漏诊率仅为 {miss_rate:.2%}！")
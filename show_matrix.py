import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# 解决 Mac 画图中文乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
print("加载数据并训练模型中...")
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. 终极模型参数
FINAL_PARAMS = {
    'n_estimators': 141,
    'max_depth': 8,
    'class_weight': {0: 1.0, 1: 5.6187},
    'random_state': 42
}
BEST_THRESHOLD = 0.5920

model = RandomForestClassifier(**FINAL_PARAMS, n_jobs=-1)
model.fit(X_train, y_train)

# 3. 预测并生成矩阵
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= BEST_THRESHOLD).astype(int)
cm = confusion_matrix(y_test, y_pred)

# 4. 打印到终端
print("\n" + "="*50)
print("📝 最终模型考核结果 (基于真实测试集)")
print("="*50)
print(f"整体 AUC 评分: {roc_auc_score(y_test, y_prob):.4f}\n")

print("【混淆矩阵】")
print(f"                 预测健康(0)     预测患病(1)")
print(f"实际健康(0)   TN: {cm[0][0]:>8}    FP: {cm[0][1]:>8}")
print(f"实际患病(1)   FN: {cm[1][0]:>8}    TP: {cm[1][1]:>8}")

miss_rate = cm[1][0] / (cm[1][0] + cm[1][1])
print(f"\n💡 核心临床指标：在真正的患病者中，模型的漏诊率仅为 {miss_rate:.2%}！")
print("\n【详细分类报告】")
print(classification_report(y_test, y_pred, target_names=["健康(0)", "患病(1)"], digits=4))

# 5. 画出一张漂亮的热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["预测健康", "预测患病"],
            yticklabels=["实际健康", "实际患病"])
plt.title("模型测试集混淆矩阵 (Confusion Matrix)")
plt.ylabel("真实标签")
plt.xlabel("模型预测")
plt.tight_layout()
plt.show()
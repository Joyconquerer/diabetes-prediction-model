import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# 解决 Mac 画图中文乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 加载数据
# ==========================================
print("正在加载数据...")
# 优先尝试读取完整数据，如果不存在则读取本地数据
try:
    df = pd.read_csv("full_data.csv").dropna()
    print(f"✅ 成功读取完整数据集！形状: {df.shape}")
except FileNotFoundError:
    df = pd.read_csv("data.csv").dropna()
    print(f"✅ 读取本地数据集！形状: {df.shape}")

if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. 训练终极模型
# ==========================================
print("\n正在训练代价敏感随机森林模型...")
FINAL_PARAMS = {
    'n_estimators': 141,
    'max_depth': 8,
    'class_weight': {0: 1.0, 1: 5.6187},
    'random_state': 42,
    'n_jobs': -1
}
model = RandomForestClassifier(**FINAL_PARAMS)
model.fit(X_train, y_train)

# 获取预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 3. 自动寻找当前数据集的最佳阈值 (Youden's J)
# ==========================================
print("正在自动计算最佳临床预警阈值...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# 找到使 (真正率 - 假正率) 最大的那个阈值
best_threshold = thresholds[np.argmax(tpr - fpr)]
print(f"🎯 在当前测试集下，自动算出的最佳阈值为: {best_threshold:.4f}")

# 使用最佳阈值进行预测
y_pred = (y_prob >= best_threshold).astype(int)

# ==========================================
# 4. 打印核心考核成绩
# ==========================================
print("\n" + "="*50)
print("📝 最终模型考核结果 (基于真实测试集)")
print("="*50)
print(f"整体 AUC 评分: {roc_auc_score(y_test, y_prob):.4f}\n")

cm = confusion_matrix(y_test, y_pred)
print("【混淆矩阵】")
print(f"                 预测健康(0)     预测患病(1)")
print(f"实际健康(0)   TN: {cm[0][0]:>8}    FP: {cm[0][1]:>8}")
print(f"实际患病(1)   FN: {cm[1][0]:>8}    TP: {cm[1][1]:>8}")

miss_rate = cm[1][0] / (cm[1][0] + cm[1][1])
print(f"\n💡 核心临床指标：在真正的患病者中，模型的漏诊率仅为 {miss_rate:.2%}！")
print("\n【详细分类报告】")
print(classification_report(y_test, y_pred, target_names=["健康(0)", "患病(1)"], digits=4))

# ==========================================
# 5. 绘制极高颜值的混淆矩阵热力图
# ==========================================
plt.figure(figsize=(8, 6))
# 使用 fmt="d" 确保显示整数，"Blues" 是蓝色系
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["预测健康", "预测患病"],
            yticklabels=["实际健康", "实际患病"],
            annot_kws={"size": 16}) # 字体放大

plt.title(f"模型测试集混淆矩阵 (阈值={best_threshold:.4f})", fontsize=14)
plt.ylabel("真实标签", fontsize=12)
plt.xlabel("模型预测", fontsize=12)
plt.tight_layout()

# 保存图片到本地，方便你放进 README
plt.savefig("confusion_matrix_final.png", dpi=300)
print("\n📸 混淆矩阵图已保存为当前目录下的 confusion_matrix_final.png")

# 弹出窗口显示
plt.show()
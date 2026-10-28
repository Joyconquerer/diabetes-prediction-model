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

print("启动自动寻优... （需要几分钟时间，去喝口水）")

# 2. 定义我们要让机器自己尝试的参数网格
n_estimators_list = [50, 100, 200]
max_depth_list = [8, 12, 16]
weight_list = [{0: 1, 1: 2}, {0: 1, 1: 5}, {0: 1, 1: 10}]

best_model = None
best_threshold = 0
best_fn = float('inf')  # 先假设漏诊无穷大
best_fp = float('inf')
best_metrics = {}

# 3. 开始自动化三重循环搜索
for n in n_estimators_list:
    for d in max_depth_list:
        for w in weight_list:
            # 训练模型
            rf = RandomForestClassifier(
                n_estimators=n, max_depth=d, class_weight=w, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_prob = rf.predict_proba(X_test)[:, 1]
            
            # 计算 ROC 曲线，寻找每个模型的最佳阈值
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            curr_threshold = thresholds[best_idx]
            
            # 预测
            y_pred = (y_prob >= curr_threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            fn = cm[1][0] # 漏诊
            fp = cm[0][1] # 误诊
            
            # 自动评估条件：我们要漏诊极低，同时误诊也别太离谱
            # 如果当前模型的漏诊比历史最佳少，或者漏诊一样但误诊更少，就更新最佳记录
            if fn < best_fn or (fn == best_fn and fp < best_fp):
                best_fn = fn
                best_fp = fp
                best_model = rf
                best_threshold = curr_threshold
                best_metrics = {
                    "n_estimators": n, "max_depth": d, "weight": w,
                    "auc": roc_auc_score(y_test, y_prob)
                }

# 4. 自动输出最终结果
print("\n" + "="*50)
print("🏆 自动搜索完成！找到最佳参数组合：")
print(f"树的数量 (n_estimators): {best_metrics['n_estimators']}")
print(f"最大深度 (max_depth): {best_metrics['max_depth']}")
print(f"惩罚权重 (class_weight): {best_metrics['weight']}")
print(f"最佳阈值: {best_threshold:.4f}")
print(f"AUC 值: {best_metrics['auc']:.4f}")
print(f"极限漏诊人数 (FN): {best_fn}")
print(f"误诊人数 (FP): {best_fp}")

print("\n【最优模型混淆矩阵】")
y_prob_best = best_model.predict_proba(X_test)[:, 1]
y_pred_best = (y_prob_best >= best_threshold).astype(int)
print(confusion_matrix(y_test, y_pred_best))
print("\n【最优模型分类报告】")
print(classification_report(y_test, y_pred_best, target_names=["无糖尿病", "有糖尿病"]))
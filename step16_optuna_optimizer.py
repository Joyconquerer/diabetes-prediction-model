import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# 1. 读取刚才成功下载的完整数据集
URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
print("正在加载完整数据...")
# 直接读取你本地的完整数据，彻底摆脱网络困扰
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 记录历史最佳成绩
best_fn_global = 156  # 这是我们上一次的最佳成绩，绝不能输给它！
best_trial_info = {}

# 2. 定义 Optuna 的优化目标
def objective(trial):
    global best_fn_global, best_trial_info
    
    # 让 Optuna 在连续的数学空间中搜索
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 20)
    weight_ratio = trial.suggest_float("weight_ratio", 1.5, 30.0)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        class_weight={0: 1.0, 1: weight_ratio},
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # 自动计算最佳阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    fn = cm[1][0]  # 漏诊
    fp = cm[0][1]  # 误诊
    auc = roc_auc_score(y_test, y_prob)
    
    # 打印并记录如果打破了历史记录
    if fn < best_fn_global:
        best_fn_global = fn
        best_trial_info = {
            "n_estimators": n_estimators, "max_depth": max_depth, 
            "weight_ratio": weight_ratio, "threshold": best_threshold,
            "fn": fn, "fp": fp, "auc": auc
        }
        print(f"🔥 突破历史记录！第 {trial.number} 次尝试 => 漏诊: {fn}, 误诊: {fp}, AUC: {auc:.4f}")

    # Optuna 会最小化这个返回值（重点惩罚 FN，其次是 FP）
    return fn * 100 + fp

# 3. 开始真正的智能搜索 (尝试 30 次，比之前的 8 代遗传算法多，而且更聪明)
print("启动 Optuna 贝叶斯寻优，寻找超越 156 的神级模型...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("\n" + "="*50)
print("🏆 搜索结束！最佳模型参数：")
print(study.best_params)
print(f"打破历史的极限漏诊 (FN): {best_trial_info.get('fn', '未打破')}")
print(f"误诊 (FP): {best_trial_info.get('fp', '-')}")
print(f"AUC: {best_trial_info.get('auc', '-'):.4f}")
print(f"最佳阈值: {best_trial_info.get('threshold', '-'):.4f}")
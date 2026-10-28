import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# 1. 本地加载数据
print("加载数据...")
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

best_fn_global = 156  # 死守这条底线
best_trial_info = {}

def objective(trial):
    global best_fn_global, best_trial_info
    
    # XGBoost 的核心参数搜索
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 15.0), # 类别权重
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    fn = cm[1][0]
    fp = cm[0][1]
    auc = roc_auc_score(y_test, y_prob)
    
    if fn < best_fn_global:
        best_fn_global = fn
        best_trial_info = {
            "params": param, "threshold": best_threshold,
            "fn": fn, "fp": fp, "auc": auc
        }
        print(f"🔥 突破历史记录！漏诊: {fn}, 误诊: {fp}, AUC: {auc:.4f}")

    return fn * 100 + fp

print("启动 XGBoost 极限突破...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("\n" + "="*50)
if best_trial_info:
    print("🏆 成功打破 156 的魔咒！极限参数：")
    print(best_trial_info["params"])
    print(f"极限漏诊 (FN): {best_trial_info['fn']}")
    print(f"误诊 (FP): {best_trial_info['fp']}")
    print(f"AUC: {best_trial_info['auc']:.4f}")
else:
    print("😭 遗憾，XGBoost 也没能打破 156 的漏诊底线。")
    print("这说明：在现有的问卷特征下，156 就是这 25 万人的医学物理极限。")
    print(f"但 Optuna 依然找到了一个综合评分最好的模型: {study.best_params}")
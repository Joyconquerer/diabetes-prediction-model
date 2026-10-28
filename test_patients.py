import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. 读取数据
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 提取你最终的模型参数
FINAL_PARAMS = {
    'n_estimators': 141,
    'max_depth': 8,
    'class_weight': {0: 1.0, 1: 5.6187},
    'random_state': 42
}
BEST_THRESHOLD = 0.5920

# 2. 用全量数据训练这个最终模型
print("正在训练最终模型...")
model = RandomForestClassifier(**FINAL_PARAMS, n_jobs=-1)
model.fit(X, y)

# ==========================================
# 3. 从测试集中抽取两个真实病例
# ==========================================
# 我们找一个真实标签是 1（有糖尿病）的病人
patient_diabetes = X[y == 1].iloc[0:1]
true_label_diabetes = y[y == 1].iloc[0]

# 我们找一个真实标签是 0（健康）的病人
patient_healthy = X[y == 0].iloc[0:1]
true_label_healthy = y[y == 0].iloc[0]

# ==========================================
# 4. 定义预测函数
# ==========================================
def predict_patient(patient_data, true_label):
    # 预测患病概率
    prob = model.predict_proba(patient_data)[0][1]
    
    # 打印病人的几个核心指标（方便你对照）
    print(f"\n--- 病例数据 ---")
    print(f"真实情况: {'有糖尿病' if true_label == 1 else '健康'}")
    print(f"BMI: {patient_data['BMI'].values[0]} | 高血压: {patient_data['HighBP'].values[0]} | 高胆固醇: {patient_data['HighChol'].values[0]} | 年龄阶段: {patient_data['Age'].values[0]}")
    
    # 模型判定
    if prob >= BEST_THRESHOLD:
        print(f"🚨 模型判定: ⚠️ 预警！患病概率 {prob:.2%}，建议复查。")
    else:
        print(f"✅ 模型判定: 正常。患病概率 {prob:.2%}，注意保持。")

# ==========================================
# 5. 让模型开始“看病”
# ==========================================
print("="*50)
print("🩺 模型正在对新病人进行诊断...")

print("\n【病人 A】")
predict_patient(patient_diabetes, true_label_diabetes)

print("\n【病人 B】")
predict_patient(patient_healthy, true_label_healthy)
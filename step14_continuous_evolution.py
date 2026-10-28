import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import random

# 1. 加载数据
df = pd.read_csv("data.csv").dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 连续空间进化参数 ---
POP_SIZE = 12          # 种群大小
GENERATIONS = 15       # 繁衍 15 代
MUTATION_RATE = 0.3    # 提高突变率

# 基因定义：现在权重和参数可以连续变化了！
def generate_random_gene():
    return {
        "n_estimators": random.randint(30, 200),
        "max_depth": random.randint(5, 20),
        # 核心改变：权重是 1.0 到 20.0 之间的随机浮点数！
        "weight_ratio": random.uniform(1.5, 20.0) 
    }

# 评估函数
def evaluate_fitness(gene):
    # 把浮点数权重传进模型
    class_weight = {0: 1.0, 1: gene["weight_ratio"]}
    
    rf = RandomForestClassifier(
        n_estimators=gene["n_estimators"], 
        max_depth=gene["max_depth"], 
        class_weight=class_weight, 
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # 找最佳阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fn = cm[1][0]
    fp = cm[0][1]
    auc = roc_auc_score(y_test, y_prob)
    
    # 适应度惩罚（漏诊 FN 罚 100，误诊 FP 罚 1）
    penalty = fn * 100 + fp
    return penalty, fn, fp, auc, best_threshold

# 初始化种群
population = [generate_random_gene() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    print(f"\n--- 第 {gen+1} 代繁衍 ---")
    
    scored_population = []
    for gene in population:
        penalty, fn, fp, auc, thres = evaluate_fitness(gene)
        scored_population.append((penalty, fn, fp, auc, thres, gene))
    
    # 按惩罚排序
    scored_population.sort(key=lambda x: x[0])
    
    best = scored_population[0]
    print(f"最优个体: 漏诊(FN)={best[1]}, 误诊(FP)={best[2]}, AUC={best[3]:.4f}")
    print(f"  基因: 树={best[5]['n_estimators']}, 深度={best[5]['max_depth']}, 权重={best[5]['weight_ratio']:.2f}")
    
    # 保留前 4 名
    elites = [item[5] for item in scored_population[:4]]
    
    next_generation = elites.copy()
    while len(next_generation) < POP_SIZE:
        p1 = random.choice(elites)
        p2 = random.choice(elites)
        
        # 交叉：随机取父母一方的特征
        child = {
            "n_estimators": random.choice([p1["n_estimators"], p2["n_estimators"]]),
            "max_depth": random.choice([p1["max_depth"], p2["max_depth"]]),
            "weight_ratio": random.choice([p1["weight_ratio"], p2["weight_ratio"]])
        }
        
        # 突变：允许微小的浮点数漂移，而不是死板的跳跃
        if random.random() < MUTATION_RATE:
            mut_type = random.choice(["n_estimators", "max_depth", "weight_ratio"])
            if mut_type == "n_estimators":
                # 在原有基础上加减 20 以内
                child["n_estimators"] = max(10, min(200, child["n_estimators"] + random.randint(-20, 20)))
            elif mut_type == "max_depth":
                child["max_depth"] = max(3, min(20, child["max_depth"] + random.randint(-2, 2)))
            else:
                # 浮点数权重突变（上下浮动 2.0，限制在 1.0-30.0 之间）
                child["weight_ratio"] = max(1.0, min(30.0, child["weight_ratio"] + random.uniform(-2.0, 2.0)))
                
        next_generation.append(child)
        
    population = next_generation

print("\n" + "="*50)
print("🧬 终极连续进化结束！最终基因：")
print(best[5])
print(f"极限漏诊 (FN): {best[1]}")
print(f"误诊 (FP): {best[2]}")
print(f"AUC: {best[3]:.4f}")
print(f"最佳阈值: {best[4]:.4f}")
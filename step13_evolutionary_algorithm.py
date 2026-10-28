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

# --- 进化参数 ---
POP_SIZE = 10          # 种群大小（每代造 10 个模型）
GENERATIONS = 10       # 繁衍代数
MUTATION_RATE = 0.2    # 基因突变率

# 基因定义
def generate_random_gene():
    return {
        "n_estimators": random.choice([30, 50, 80, 100, 150]),
        "max_depth": random.choice([6, 8, 10, 12, 15]),
        "weight": random.choice([{0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 8}, {0: 1, 1: 10}, {0: 1, 1: 15}])
    }

# 适应度评估函数（核心：FN 越小越好，FN 相同时 FP 越小越好）
def evaluate_fitness(gene):
    rf = RandomForestClassifier(
        n_estimators=gene["n_estimators"], 
        max_depth=gene["max_depth"], 
        class_weight=gene["weight"], 
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # 找最佳阈值（Youden's J）
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fn = cm[1][0]
    fp = cm[0][1]
    auc = roc_auc_score(y_test, y_prob)
    
    # 适应度得分：漏诊权重极大，误诊次之。分数越高越好。
    # 我们可以直接用 (FN * 100 + FP) 作为惩罚，惩罚越低适应度越高
    penalty = fn * 100 + fp
    return penalty, fn, fp, auc, best_threshold

# 2. 初始化种群
population = [generate_random_gene() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    print(f"\n--- 第 {gen+1} 代繁衍开始 ---")
    
    # 评估种群适应度
    scored_population = []
    for gene in population:
        penalty, fn, fp, auc, thres = evaluate_fitness(gene)
        scored_population.append((penalty, fn, fp, auc, thres, gene))
    
    # 按惩罚从小到大排序（越小越好）
    scored_population.sort(key=lambda x: x[0])
    
    # 打印当前代最优秀的个体
    best = scored_population[0]
    print(f"当前代最优: 漏诊(FN)={best[1]}, 误诊(FP)={best[2]}, AUC={best[3]:.4f}, 深度={best[5]['max_depth']}, 权重={best[5]['weight']}")
    
    # 3. 选择精英（保留前 4 名）
    elites = [item[5] for item in scored_population[:4]]
    
    # 4. 繁衍下一代
    next_generation = elites.copy()  # 精英直接晋级
    while len(next_generation) < POP_SIZE:
        # 随机选两个父母交配
        parent1 = random.choice(elites)
        parent2 = random.choice(elites)
        
        # 基因交叉（混合父母的特性）
        child = {
            "n_estimators": random.choice([parent1["n_estimators"], parent2["n_estimators"]]),
            "max_depth": random.choice([parent1["max_depth"], parent2["max_depth"]]),
            "weight": random.choice([parent1["weight"], parent2["weight"]])
        }
        
        # 基因突变
        if random.random() < MUTATION_RATE:
            mutation_type = random.choice(["n_estimators", "max_depth", "weight"])
            if mutation_type == "n_estimators":
                child["n_estimators"] = random.choice([30, 50, 80, 100, 150, 200])
            elif mutation_type == "max_depth":
                child["max_depth"] = random.choice([6, 8, 10, 12, 15, 18])
            else:
                child["weight"] = random.choice([{0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 20}])
                
        next_generation.append(child)
        
    population = next_generation

print("\n" + "="*50)
print("🧬 进化结束！最终的超级模型基因：")
print(best[5])
print(f"极限漏诊 (FN): {best[1]}")
print(f"误诊 (FP): {best[2]}")
print(f"AUC: {best[3]:.4f}")
print(f"最佳阈值: {best[4]:.4f}")
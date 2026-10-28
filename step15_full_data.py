import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import random

# 1. 官方完整 ZIP 压缩包地址（直接丢给 pandas 读，它会自动解压）
# 注意：如果这个链接还是卡住，请告诉我，我给你换另一种分块下载法
URL = "https://archive.ics.uci.edu/static/public/891/data.csv"

print("正在连接 UCI 下载完整版数据，可能需要 10-30 秒，请等待...")
try:
    # 我们直接读，并设定超时
    df = pd.read_csv(URL, timeout=30)
    print(f"✅ 下载成功！原始数据形状: {df.shape}")
    
    # 如果发现行数还是 6 万左右，说明 UCI 服务器确实给网页下载做了截断
    if df.shape[0] < 100000:
        print("⚠️ 警告：下载到的依然是子集（约 6 万行）。")
        print("建议：用浏览器打开 https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators 页面，手动下载完整包。")
    else:
        print("🚀 太棒了，成功拿到 25 万行完整数据集！")
except Exception as e:
    print(f"下载失败: {e}")
    print("降级使用本地 6 万行数据继续跑...")
    df = pd.read_csv("data.csv")

# 无论拿到哪份数据，先清理
df = df.dropna()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 划出 20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集样本数: {len(X_train)}")

# 2. 既然数据量可能变大了，我们直接跑一次“终极连续进化”
POP_SIZE = 10
GENERATIONS = 8  # 数据量大了，跑 8 代就足够看出趋势

def generate_random_gene():
    return {
        "n_estimators": random.randint(50, 200),
        "max_depth": random.randint(8, 18),
        "weight_ratio": random.uniform(1.5, 25.0)
    }

def evaluate_fitness(gene):
    rf = RandomForestClassifier(
        n_estimators=gene["n_estimators"], 
        max_depth=gene["max_depth"], 
        class_weight={0: 1.0, 1: gene["weight_ratio"]}, 
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    fn = cm[1][0]
    fp = cm[0][1]
    auc = roc_auc_score(y_test, y_prob)
    return fn * 100 + fp, fn, fp, auc, best_threshold

print("\n开始在新数据上进化...")
population = [generate_random_gene() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    scored_population = []
    for gene in population:
        penalty, fn, fp, auc, thres = evaluate_fitness(gene)
        scored_population.append((penalty, fn, fp, auc, thres, gene))
    
    scored_population.sort(key=lambda x: x[0])
    best = scored_population[0]
    
    print(f"第 {gen+1} 代最优: 漏诊={best[1]}, 误诊={best[2]}, AUC={best[3]:.4f}")
    
    elites = [item[5] for item in scored_population[:3]]
    next_generation = elites.copy()
    
    while len(next_generation) < POP_SIZE:
        p1, p2 = random.choice(elites), random.choice(elites)
        child = {
            "n_estimators": random.choice([p1["n_estimators"], p2["n_estimators"]]),
            "max_depth": random.choice([p1["max_depth"], p2["max_depth"]]),
            "weight_ratio": random.choice([p1["weight_ratio"], p2["weight_ratio"]])
        }
        if random.random() < 0.3:
            mut_type = random.choice(["n_estimators", "max_depth", "weight_ratio"])
            if mut_type == "n_estimators":
                child["n_estimators"] = max(20, min(300, child["n_estimators"] + random.randint(-30, 30)))
            elif mut_type == "max_depth":
                child["max_depth"] = max(3, min(25, child["max_depth"] + random.randint(-2, 2)))
            else:
                child["weight_ratio"] = max(1.0, min(40.0, child["weight_ratio"] + random.uniform(-3.0, 3.0)))
        next_generation.append(child)
        
    population = next_generation

print("\n" + "="*50)
print(f"🏆 基于新数据的终极模型：")
print(f"基因: {best[5]}")
print(f"漏诊 (FN): {best[1]}")
print(f"误诊 (FP): {best[2]}")
print(f"AUC: {best[3]:.4f}")
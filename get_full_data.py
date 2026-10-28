import pandas as pd
print("正在下载完整数据集，可能需要几十秒...")
url = "https://archive.ics.uci.edu/static/public/891/data.csv"
df = pd.read_csv(url)
# 保存为完整的本地文件
df.to_csv("full_data.csv", index=False)
print(f"✅ 下载完成！完整数据形状: {df.shape}")
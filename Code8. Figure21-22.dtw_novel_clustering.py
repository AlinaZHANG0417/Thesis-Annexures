# dtw_novel_clustering.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号


# ---------- 参数 ----------
data_path = "H:/ZHANGJINGYI-20250330/data/2.句子情感数据/B01-B18句子情感满意原始数据/2.归一化数据/All_Novels_Emotion_Curves_LOESS_Interpolated.csv"
output_dir = "./DTW_Clustering_Results"
n_clusters = 4  # 聚类数可根据你实际情况调整

os.makedirs(output_dir, exist_ok=True)

# ---------- 读取数据 ----------
df = pd.read_csv(data_path)
novel_ids = df['novel'].unique()
novel_curves = []

for novel in novel_ids:
    series = df[df['novel'] == novel]['emotion_score'].values
    novel_curves.append(series)

X = np.array(novel_curves)  # (18, 100)

# ---------- 聚类 ----------
print("开始聚类 ...")
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
labels = model.fit_predict(X)

# ---------- 保存标签 ----------
label_df = pd.DataFrame({'novel': novel_ids, 'cluster': labels})
label_df.to_csv(os.path.join(output_dir, "novel_cluster_labels.csv"), index=False)
print("聚类标签已保存。")

# ---------- 平均曲线图 ----------
print("绘制每类平均情感曲线 ...")
cluster_avg_curves = []
for cluster_id in range(n_clusters):
    cluster_data = X[labels == cluster_id]
    mean_curve = np.mean(cluster_data, axis=0)
    cluster_avg_curves.append(mean_curve)

avg_curves_df = pd.DataFrame()
for idx, curve in enumerate(cluster_avg_curves):
    temp_df = pd.DataFrame({
        'progress': np.linspace(0, 1, len(curve)),
        'emotion_score': curve,
        'cluster': f'Cluster {idx}'
    })
    avg_curves_df = pd.concat([avg_curves_df, temp_df], ignore_index=True)

plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_curves_df, x='progress', y='emotion_score', hue='cluster')
plt.title('每类小说的平均情感曲线')
plt.savefig(os.path.join(output_dir, "average_curves_per_cluster.png"))
plt.close()

# ---------- t-SNE 投影图 ----------
# ---------- t-SNE 投影图（修复版） ----------
print("计算 t-SNE ...")
distance_matrix = np.array([[dtw(a, b) for b in X] for a in X])
tsne = TSNE(n_components=2, metric="precomputed", perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(distance_matrix)

tsne_df = pd.DataFrame({
    'x': tsne_result[:, 0],
    'y': tsne_result[:, 1],
    'novel': novel_ids,
    'cluster': [f'Cluster {i}' for i in labels]
})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x='x', y='y', hue='cluster', style='cluster', s=100)
for _, row in tsne_df.iterrows():
    plt.text(row['x'] + 0.3, row['y'], row['novel'], fontsize=9)
plt.title('小说情感曲线的 t-SNE 投影')
plt.savefig(os.path.join(output_dir, "tsne_projection.png"))
plt.close()

# ---------- 所有情感曲线（按聚类结果排序） ----------
print("绘制所有小说情感曲线图（按聚类分组）...")
df['cluster'] = df['novel'].map(lambda x: f"Cluster {labels[np.where(novel_ids == x)[0][0]]}")
df['novel_sorted'] = df['novel'].map(lambda x: f"{labels[np.where(novel_ids == x)[0][0]]}-{x}")

plt.figure(figsize=(14, 8))
sns.lineplot(data=df, x='progress', y='emotion_score', hue='novel_sorted', style='cluster', palette='tab20')
plt.title('所有小说情感曲线（按聚类排序）')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_novels_curves_by_cluster.png"))
plt.close()

print(f"全部分析完成！结果已保存至：{output_dir}")

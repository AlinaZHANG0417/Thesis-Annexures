
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import argrelextrema
import os

# === 参数配置 ===
input_folder = r"F:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据"
output_folder = r"F:\ZHANGJINGYI-20250330\data\4.原始、移平、LOESS平滑曲线图\loess平滑曲线_自动节点_阈值合并"
os.makedirs(output_folder, exist_ok=True)

# 分段颜色（循环使用）
segment_colors = ['#f0e68c', '#c2dfff', '#ffd8b1', '#d9ead3', '#f4cccc']

# 阈值设置：控制“保留多大情绪波动”的拐点（可根据需要调整）
AMPLITUDE_THRESHOLD = 0.2  # ✔️ 设置的位置（默认值为0.2）

# 基于幅度合并节点函数
def detect_significant_inflections(smoothed, threshold=AMPLITUDE_THRESHOLD):
    maxima = argrelextrema(smoothed, np.greater)[0]
    minima = argrelextrema(smoothed, np.less)[0]
    candidates = sorted(np.concatenate((maxima, minima)))

    if len(candidates) < 2:
        return []

    # 计算情绪波动幅度（两点之间的值差）
    amplitudes = [abs(smoothed[candidates[i]] - smoothed[candidates[i - 1]])
                  for i in range(1, len(candidates))]

    # 保留满足阈值条件的拐点（中间点）
    valid_nodes = []
    for i in range(1, len(candidates)):
        if amplitudes[i - 1] >= threshold:
            valid_nodes.append(candidates[i - 1])

    # 也保留最后一个节点作为段尾参考（但不重复）
    if len(candidates) > 1 and candidates[-1] not in valid_nodes:
        if abs(smoothed[candidates[-1]] - smoothed[candidates[-2]]) >= threshold:
            valid_nodes.append(candidates[-1])

    return sorted(list(set(valid_nodes)))

# 主绘图函数
def plot_loess_sentiment_with_threshold(book_id, scores, x, save_path, loess_frac=0.3, manual_nodes=None):
    smoothed = lowess(scores, x, frac=loess_frac)[:, 1]
    auto_nodes = detect_significant_inflections(smoothed, threshold=AMPLITUDE_THRESHOLD)

    # 使用手动节点（如指定）代替自动节点
    if manual_nodes is not None:
        nodes = manual_nodes
    else:
        nodes = auto_nodes

    max_peak_idx = np.argmax(smoothed)
    min_valley_idx = np.argmin(smoothed)

    plt.figure(figsize=(14, 5))
    plt.plot(x, smoothed, color='green', linewidth=2, label='LOESS Curve')
    plt.fill_between(x, 0, smoothed, where=smoothed >= 0, interpolate=True, color='green', alpha=0.2)
    plt.fill_between(x, 0, smoothed, where=smoothed < 0, interpolate=True, color='red', alpha=0.2)

    segment_edges = [0] + nodes + [len(x)]
    for i in range(len(segment_edges) - 1):
        plt.axvspan(segment_edges[i], segment_edges[i + 1],
                    facecolor=segment_colors[i % len(segment_colors)], alpha=0.2)

    for node in nodes:    # 在当前节点 y 值基础上再偏移一段距离
        y_value = smoothed[node]
        offset = (max(smoothed) - min(smoothed)) * 0.1  # 可调节的偏移量
        y_position = y_value - offset if y_value > 0 else y_value + offset

        plt.axvline(x=node, color='red', linestyle='--', alpha=0.8, linewidth=1.2)
        plt.text(node, y_position, f"{node}", color='red',
                 fontsize=10, ha='center', va='bottom')


    # Climax & Valley
    for idx, label in zip([max_peak_idx, min_valley_idx], ['Climax', 'Valley']):
        plt.axvline(x=idx, color='blue', linestyle='--', linewidth=1.2)
        plt.text(idx, smoothed[idx], label, color='blue', fontsize=11,
                 ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))

    for i in range(len(segment_edges) - 1):
        mid = (segment_edges[i] + segment_edges[i + 1]) // 2
        height = max(smoothed) * 0.7
        plt.text(mid, height, f"Seg {i + 1}", fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), ha='center')

    plt.title(f"{book_id} | Emotional Trend (LOESS) with Thresholded Nodes", fontsize=16)
    plt.xlabel("Sentence Index", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# 批量绘图主程序
book_ids = [f"B{str(i).zfill(2)}" for i in range(1, 19)]

manual_node_dict = {
    "B04": "valley",
    "B08": "valley"
}

for book_id in book_ids:
    file_path = os.path.join(input_folder, f"{book_id}情感汇总_处理后.csv")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue
    try:
        data = pd.read_csv(file_path)
        scores = data['Intensity_polarized'].fillna(0).values
        x = np.arange(len(scores))
        smoothed = lowess(scores, x, frac=0.3)[:, 1]

        # 自动节点先提取
        auto_nodes = detect_significant_inflections(smoothed, threshold=AMPLITUDE_THRESHOLD)

        # 检查是否需要插入valley
        if book_id in ["B04", "B08"]:
            valley_idx = int(np.argmin(smoothed))
            # 如果valley节点不在原节点列表中，再添加进去
            if valley_idx not in auto_nodes:
                auto_nodes.append(valley_idx)
                auto_nodes = sorted(auto_nodes)

        save_path = os.path.join(output_folder, f"{book_id}_LOESS_Segmented_Thresholded_Peaks.png")
        plot_loess_sentiment_with_threshold(book_id, scores, x, save_path, manual_nodes=auto_nodes)
        print(f"✅ Completed: {book_id}")
    except Exception as e:
        print(f"⚠️ Error processing {book_id}: {e}")



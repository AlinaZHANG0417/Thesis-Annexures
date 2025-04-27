import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import os

# 配置字体（不需要中文字体时可忽略中文字体设置）
rcParams['font.sans-serif'] = ['Arial']  # 使用通用字体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 文件路径
file_path = r"H:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据\情感极性汇总.xlsx"

# 读取数据
data = pd.read_excel(file_path)

# 获取样本书编号和对应的情感得分
sample_ids = data['小说编号'].unique()
positive_scores = []
negative_scores = []
neutral_scores = []

# 计算每本书的三种情感极性得分平均值
for sample_id in sample_ids:
    sample_data = data[data['小说编号'] == sample_id]
    positive_scores.append(sample_data['Positive_Polarity'].mean())
    negative_scores.append(sample_data['Negative_Polarity'].mean())
    neutral_scores.append(sample_data['Neutral_Polarity'].mean())

# 构建情感光环图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# 环形图数据
num_samples = len(sample_ids)
theta = np.linspace(0, 2 * np.pi, num_samples + 1)

# 添加间隙（样本书之间的白色边界）
gap = 0.02  # 间隙宽度

for i in range(num_samples):
    # 每本书的起始角度和宽度
    start_angle = theta[i]
    width = theta[i + 1] - start_angle - gap  # 减去间隙宽度

    # 获取当前样本的情感得分
    total = positive_scores[i] + negative_scores[i] + neutral_scores[i]
    pos_width = width * (positive_scores[i] / total)
    neg_width = width * (negative_scores[i] / total)
    neu_width = width * (neutral_scores[i] / total)

    # 绘制积极、消极、中性情感（移除边界线）
    ax.bar(start_angle, 1, width=pos_width, color="red", align='edge')
    ax.bar(start_angle + pos_width, 1, width=neg_width, color="blue", align='edge')
    ax.bar(start_angle + pos_width + neg_width, 1, width=neu_width, color="gray", align='edge')

# 添加样本书编号标签
for i, sample_id in enumerate(sample_ids):
    angle = (theta[i] + theta[i + 1]) / 2
    ax.text(angle, 1.1, sample_id, ha='center', va='center', fontsize=10, fontweight='bold')

# 绘制环中心的整体情感比例
overall_positive = sum(positive_scores)
overall_negative = sum(negative_scores)
overall_neutral = sum(neutral_scores)
sizes = [overall_positive, overall_negative, overall_neutral]
labels = ["Positive", "Negative", "Neutral"]
colors = ["red", "blue", "gray"]

# 使用 inset_axes 确保内环居中
inner_ax = ax.inset_axes([0.3, 0.3, 0.4, 0.4])  # [x, y, width, height] 相对于父图
inner_ax.pie(sizes, labels=labels, colors=colors, startangle=90,
             wedgeprops=dict(width=0.3, edgecolor='w'), textprops={'fontsize': 10})
inner_ax.set_aspect('equal')  # 保证饼图为正圆

# 标题（放在图的下方）
fig.text(0.5, 0.02, "18 Samples - Emotional Spectrum Donut Chart", ha='center', va='center', fontsize=14, fontweight='bold')

# 隐藏极坐标的默认刻度
ax.set_xticks([])
ax.set_yticks([])

# 保存图像
output_folder = r"H:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据\各样本书情感分布图"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "18_samples_emotional_spectrum.png")
plt.savefig(output_file)  # 保存最终图像
plt.close()  # 关闭图形，释放内存

print(f"情感光环图已生成并保存至：{output_file}")

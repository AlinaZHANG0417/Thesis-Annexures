import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib

# 设置中文字体（SimHei为黑体，适用于Windows系统）
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置文件路径
file_path = r'F:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据\data summary.xlsx'

# 读取数据
df = pd.read_excel(file_path)

# 定义情感强度等级
def classify_intensity(x):
    if 1 <= x <= 3:
        return '弱 (1-3)'
    elif 4 <= x <= 6:
        return '中 (4-6)'
    elif 7 <= x <= 9:
        return '强 (7-9)'
    else:
        return '未知'

df['强度等级'] = df['情感强度'].apply(classify_intensity)

# ==========================================
# 一、整体情感强度堆积柱状图和环形图
# ==========================================

# 统计每本小说每个强度等级的数量
grouped = df.groupby(['小说编号', '强度等级']).size().unstack(fill_value=0)

# 计算每本小说各等级占比
proportion = grouped.div(grouped.sum(axis=1), axis=0)

# 可视化1：堆积柱状图
plt.figure(figsize=(14, 7))
proportion[['弱 (1-3)', '中 (4-6)', '强 (7-9)']].plot(
    kind='bar',
    stacked=True,
    colormap='Pastel1',
    figsize=(14, 7),
    edgecolor='black'
)
plt.title('18本小说句子情感强度等级分布（堆积柱状图）', fontsize=16)
plt.xlabel('小说编号', fontsize=12)
plt.ylabel('比例', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='情感强度等级', fontsize=10)
plt.tight_layout()
plt.savefig(r'F:\ZHANGJINGYI-20250330\句子情感强度堆积柱状图.png')
plt.show()

# 可视化2：整体情感强度等级环形图
total_counts = df['强度等级'].value_counts().reindex(['弱 (1-3)', '中 (4-6)', '强 (7-9)'])

plt.figure(figsize=(6, 6))
colors = ['#c6dbef', '#9ecae1', '#6baed6']
plt.pie(
    total_counts,
    labels=total_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    wedgeprops=dict(width=0.4)  # 环形图
)
plt.title('18本小说整体情感强度等级分布（环形图）', fontsize=14)
plt.savefig(r'F:\ZHANGJINGYI-20250330\句子情感强度环形图.png')
plt.show()

# ==========================================
# 二、按情感极性进行强度等级分析（新增）
# ==========================================

# 统计每种情感极性下不同强度等级的数量
polar_group = df.groupby(['情感极性', '强度等级']).size().reset_index(name='数量')

# 排序用：定义强度等级为有序类别
polar_group['强度等级'] = pd.Categorical(
    polar_group['强度等级'],
    categories=['弱 (1-3)', '中 (4-6)', '强 (7-9)'],
    ordered=True
)

# 可选：将极性换为中文更清晰
polar_group['情感极性'] = polar_group['情感极性'].map({'正': '积极', '中': '中性', '负': '消极'})

# 方法1：分面柱状图
g = sns.FacetGrid(polar_group, col='情感极性', sharey=True, height=4, aspect=1)
g.map_dataframe(sns.barplot, x='强度等级', y='数量', palette='Blues_d')
g.set_axis_labels("情感强度等级", "句子数量")
g.set_titles(col_template="{col_name}情感")
plt.suptitle('不同情感极性下的情感强度等级分布（分面柱状图）', y=1.05, fontsize=16)
plt.tight_layout()
plt.savefig(r'H:\ZHANGJINGYI-20250330\极性_情感强度_分面柱状图.png')
plt.show()

# 方法2：多条折线图
line_df = polar_group.pivot(index='强度等级', columns='情感极性', values='数量').fillna(0)

plt.figure(figsize=(8, 6))
for col in line_df.columns:
    plt.plot(line_df.index, line_df[col], marker='o', label=col)

plt.title('不同情感极性下的情感强度分布趋势（折线图）', fontsize=14)
plt.xlabel('情感强度等级')
plt.ylabel('句子数量')
plt.legend(title='情感极性')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(r'F:\ZHANGJINGYI-20250330\极性_情感强度_折线图.png')
plt.show()

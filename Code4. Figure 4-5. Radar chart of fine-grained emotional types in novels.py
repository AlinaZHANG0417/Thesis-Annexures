import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置英文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
file_path = r'H:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据\data summary.xlsx'
save_dir = r'H:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据'

# 读取数据
df = pd.read_excel(file_path)

# 小说编号映射为类型
type_map = {
    'B01': 'Animal/Nature', 'B02': 'Animal/Nature', 'B03': 'Animal/Nature',
    'B04': 'Animal/Nature', 'B05': 'Animal/Nature', 'B17': 'Animal/Nature',
    'B07': 'Fantasy/Adventure', 'B10': 'Fantasy/Adventure', 'B13': 'Fantasy/Adventure',
    'B15': 'Fantasy/Adventure', 'B16': 'Fantasy/Adventure', 'B18': 'Fantasy/Adventure',
    'B06': 'Growth/Family', 'B08': 'Growth/Family', 'B09': 'Growth/Family',
    'B11': 'Growth/Family', 'B12': 'Growth/Family', 'B14': 'Growth/Family',
}
df['Novel Type'] = df['小说编号'].map(type_map)
df = df[df['Novel Type'].notnull()]

# ✅ 取绝对值后聚合
df['Abs_Intensity'] = df['Intensity_polarized'].abs()
emotion_avg = df.groupby(['Novel Type', 'Emotional Types'])['Abs_Intensity'].mean().reset_index()
pivot_df = emotion_avg.pivot(index='Novel Type', columns='Emotional Types', values='Abs_Intensity').fillna(0)

# 固定情感顺序
emotion_order = ['Joy','Tru','Sat','Hop','Grat','Fear','Sad','Disg','Anx','Ang','Disap','Pri','Sha','Calm','Surp']
pivot_df = pivot_df[emotion_order]
type_order = ['Fantasy/Adventure', 'Animal/Nature', 'Growth/Family']


# 保存数据
output_excel_path = os.path.join(save_dir, '细粒度情感类型_各小说类型_显著性均值.xlsx')
pivot_df.to_excel(output_excel_path)

# 🔄 绘图准备
labels = pivot_df.columns.tolist()
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

colors = ['#1f77b4', '#2ca02c', '#d62728']

# ======================
# 1️⃣ 三类小说类型合并雷达图
# ======================
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
for idx, (novel_type, row) in enumerate(pivot_df.iterrows()):
    values = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, values, label=novel_type, linewidth=2, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Emotional Salience Profile: All Novel Types", fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '雷达图_三类小说类型_显著性对比.png'))
plt.close()

# ======================
# 2️⃣ 各类小说单独雷达图
# ======================
# ======================
# 2️⃣ 各类小说单独雷达图（每张图画6本书）
# ======================

# 获取每本小说在每种情感类型下的绝对值平均
novel_emotion_df = df.groupby(['小说编号', 'Emotional Types'])['Abs_Intensity'].mean().reset_index()
novel_emotion_pivot = novel_emotion_df.pivot(index='小说编号', columns='Emotional Types', values='Abs_Intensity').fillna(0)

# 保证情感类型顺序一致
novel_emotion_pivot = novel_emotion_pivot[emotion_order]

# 添加小说类型列
novel_emotion_pivot['Novel Type'] = novel_emotion_pivot.index.map(lambda x: type_map.get(x, None))
novel_emotion_pivot = novel_emotion_pivot[novel_emotion_pivot['Novel Type'].notnull()]

# 分别绘制每类的雷达图（每类6本小说，每张图6条线）
for novel_type in type_order:
    subset = novel_emotion_pivot[novel_emotion_pivot['Novel Type'] == novel_type].drop(columns=['Novel Type'])

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
    for idx, (novel_id, row) in enumerate(subset.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, label=novel_id, linewidth=1.8)
        ax.fill(angles, values, alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"Emotional Salience Profiles: {novel_type.replace('/', ' and ')}", fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    plt.tight_layout()

    filename = f"雷达图_{novel_type.replace('/', '_')}_每本小说显著性.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


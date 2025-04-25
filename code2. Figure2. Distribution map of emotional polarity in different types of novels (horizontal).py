import pandas as pd
import matplotlib.pyplot as plt

# 本地路径设置
input_file = r'H:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据\情感极性汇总.xlsx'
output_file = r'H:\ZHANGJINGYI-20250330\data\2.句子情感数据\emotion_polarity_by_book_stacked_horizontal.png'

# 读取数据
df = pd.read_excel(input_file)

# 设置每本小说的标签为：类型缩写+编号，如 "F1", "A1", "G1"
type_map = {
    'Fantasy/Adventure': 'F',
    'Animal/Nature': 'A',
    'Growth/Family': 'G'
}

df['Label'] = df.groupby('Types of Novels').cumcount() + 1
df['GroupLabel'] = df['Types of Novels'].map(type_map) + df['Label'].astype(str)

# 保证按照类型顺序排列：F → A → G
type_order = ['Fantasy/Adventure', 'Animal/Nature', 'Growth/Family']
df['TypeOrder'] = df['Types of Novels'].apply(lambda x: type_order.index(x) if pd.notna(x) and x in type_order else -1)
df = df.sort_values(by=['TypeOrder', 'Label'])

# 绘制横向堆积条形图
plt.figure(figsize=(10, 8))
bars = plt.barh(df['GroupLabel'], df['Neutral_Polarity'],
                color='cornflowerblue', label='Neutral')
bars2 = plt.barh(df['GroupLabel'], df['Positive_Polarity'],
                 left=df['Neutral_Polarity'], color='darkorange', label='Positive')
bars3 = plt.barh(df['GroupLabel'],
                 df['Negative_Polarity'],
                 left=df['Neutral_Polarity'] + df['Positive_Polarity'],
                 color='mediumseagreen', label='Negative')

# 添加分组的水平分隔线与提示线
for i in [6, 12]:
    plt.axhline(y=i - 0.5, color='gray', linestyle='--', linewidth=1)

# 设置标题与标签
plt.title("Emotional Polarity Distribution of 18 Novels by Type", fontsize=13)
plt.ylabel("Books (Grouped by Type: F=Fantasy, A=Animal, G=Growth)")
plt.xlabel("Polarity Value")
plt.legend(loc='lower right')
plt.tight_layout()

# 保存图像
plt.savefig(output_file, dpi=300)
plt.show()
print(f"图像已保存到本地：{output_file}")

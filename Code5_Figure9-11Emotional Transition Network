import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建输出目录
base_output_path = r"H:\ZHANGJINGYI-20250330\emotion_network_analysis\Emotion_Network_Output"
os.makedirs(base_output_path, exist_ok=True)

# 为每个小说类型单独创建子目录
novel_types = ['Animal', 'Fantasy_Adventure', 'Growth_Family']
for nt in novel_types:
    os.makedirs(os.path.join(base_output_path, nt), exist_ok=True)

# 设置输出路径
output_folder = r"H:\ZHANGJINGYI-20250330\emotion_network_analysis\Emotion_Network_Output"
os.makedirs(output_folder, exist_ok=True)

# 读取数据
data_path = r"H:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据\data summary.xlsx"
df = pd.read_excel(data_path)

# 小说类型映射
type_map = {
    'B01': 'Animal', 'B02': 'Animal', 'B03': 'Animal',
    'B04': 'Animal', 'B05': 'Animal', 'B17': 'Animal',
    'B07': 'Fantasy_Adventure', 'B10': 'Fantasy_Adventure', 'B13': 'Fantasy_Adventure',
    'B15': 'Fantasy_Adventure', 'B16': 'Fantasy_Adventure', 'B18': 'Fantasy_Adventure',
    'B06': 'Growth_Family', 'B08': 'Growth_Family', 'B09': 'Growth_Family',
    'B11': 'Growth_Family', 'B12': 'Growth_Family', 'B14': 'Growth_Family',
}

df['Novel_Type'] = df['小说编号'].map(type_map)

# 去除NA节点
df = df[df['Emotional Types'] != 'NA']

# 定义阈值（过滤频数小于阈值的转移）
threshold = 5

# 对每一小说类型进行单独处理
for novel_type, group_df in df.groupby('Novel_Type'):

    # 计算情感转移频数
    transitions = {}
    for novel_id, novel_group in group_df.groupby('小说编号'):
        emotions = novel_group.sort_values(by='句子编号')['Emotional Types'].tolist()
        for i in range(len(emotions)-1):
            pair = (emotions[i], emotions[i+1])
            transitions[pair] = transitions.get(pair, 0) + 1

    # 应用阈值过滤
    transitions_filtered = {k: v for k, v in transitions.items() if v >= threshold}

    # 构建网络
    G = nx.DiGraph()

    for (src, dst), weight in transitions_filtered.items():
        G.add_edge(src, dst, weight=weight)

    # 计算节点中心性
    out_degree = dict(G.out_degree(weight='weight'))
    in_degree = dict(G.in_degree(weight='weight'))
    betweenness = nx.betweenness_centrality(G, weight='weight')

    centrality_df = pd.DataFrame({
        'Emotion': list(G.nodes()),
        'OutDegree': [out_degree.get(n, 0) for n in G.nodes()],
        'InDegree': [in_degree.get(n, 0) for n in G.nodes()],
        'Betweenness': [betweenness.get(n, 0) for n in G.nodes()],
    }).sort_values(by='Betweenness', ascending=False)

    # 保存中心性数据
    subfolder = os.path.join(output_folder, novel_type)
    centrality_path = os.path.join(subfolder, f"{novel_type}_Emotion_Centrality.xlsx")
    transition_path = os.path.join(subfolder, f"{novel_type}_Transition_Probability_Matrix.xlsx")
    network_path = os.path.join(subfolder, f"{novel_type}_Emotion_Transition_Network.png")

    centrality_df.to_excel(centrality_path, index=False)
    
    # 导出 Gephi 边列表格式 CSV：Source, Target, Weight
    edge_list_df = pd.DataFrame([
        {'Source': src, 'Target': dst, 'Weight': weight}
        for (src, dst), weight in transitions_filtered.items()
    ])

    # 保存到对应子文件夹
    edge_list_path = os.path.join(subfolder, f"{novel_type}_Emotion_Edge_List.csv")
    edge_list_df.to_csv(edge_list_path, index=False, encoding='utf-8-sig')

    # 构建转移概率矩阵
    emotions_list = list(G.nodes())
    transition_matrix = pd.DataFrame(0, index=emotions_list, columns=emotions_list)

    for (src, dst), weight in transitions_filtered.items():
        transition_matrix.loc[src, dst] = weight

    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    transition_matrix.fillna(0, inplace=True)

    # 保存转移概率矩阵
    matrix_path = os.path.join(output_folder, f"{novel_type}_Transition_Probability_Matrix.xlsx")
    transition_matrix.to_excel(matrix_path)

    # 绘制情感网络图
        # 重新绘制图像，优化可视效果
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2, seed=42)

    # 颜色：高亮中介中心性排名前3的节点
    top_nodes = centrality_df.head(3)['Emotion'].tolist()
    node_colors = ['orange' if node in top_nodes else 'skyblue' for node in G.nodes()]
    node_sizes = [3000 if node in top_nodes else 2000 for node in G.nodes()]

    # 边权重转为边粗细
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [1 + np.log(w) for w in edge_weights]

    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrows=True, alpha=0.6, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

    # 绘制边权数字
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

    plt.title(f"Emotion Transition Network ({novel_type})", fontsize=18)
    plt.axis('off')
    plt.tight_layout()

    # 保存图像
    network_path = os.path.join(output_folder, f"{novel_type}_Emotion_Transition_Network.png")
    plt.savefig(network_path, dpi=400)
    plt.close()


print("Processing complete. Outputs saved to:", output_folder)

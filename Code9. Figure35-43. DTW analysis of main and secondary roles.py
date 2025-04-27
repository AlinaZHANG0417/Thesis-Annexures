
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from statsmodels.nonparametric.smoothers_lowess import lowess

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件夹路径配置（请根据你本地实际路径修改）
main_role_folder = r"H:\ZHANGJINGYI-20250330\data\5.提取角色0330\主要角色\主要角色归一化后"
minor_role_folder = r"H:\ZHANGJINGYI-20250330\data\5.提取角色0330\次要角色\次要角色归一化后"
full_book_path = r"H:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据\data summary.xlsx"
output_folder = r"H:\ZHANGJINGYI-20250330\data\DTW输出图表2"

os.makedirs(output_folder, exist_ok=True)

type_map = {
    'B01': 'Animal/Nature', 'B02': 'Animal/Nature', 'B03': 'Animal/Nature',
    'B04': 'Animal/Nature', 'B05': 'Animal/Nature', 'B17': 'Animal/Nature',
    'B07': 'Fantasy/Adventure', 'B10': 'Fantasy/Adventure', 'B13': 'Fantasy/Adventure',
    'B15': 'Fantasy/Adventure', 'B16': 'Fantasy/Adventure', 'B18': 'Fantasy/Adventure',
    'B06': 'Growth/Family', 'B08': 'Growth/Family', 'B09': 'Growth/Family',
    'B11': 'Growth/Family', 'B12': 'Growth/Family', 'B14': 'Growth/Family',
}

full_df = pd.read_excel(full_book_path)
full_data_grouped = full_df.groupby('小说编号')

results = []

for i in range(1, 19):
    book_id = f'B{i:02d}'
    if book_id not in type_map:
        continue

    novel_type = type_map[book_id]

    main_path = os.path.join(main_role_folder, f"{book_id}主要_归一化处理后.csv")
    minor_path = os.path.join(minor_role_folder, f"{book_id}次要_归一化处理后.csv")

    if not os.path.exists(main_path) or not os.path.exists(minor_path):
        print(f"缺失文件：{book_id}")
        continue

    main_df = pd.read_csv(main_path, encoding='utf-8')
    minor_df = pd.read_csv(minor_path, encoding='utf-8')
    full_series = full_data_grouped.get_group(book_id)['Intensity_polarized'].reset_index(drop=True)

    def smooth(series, frac=0.05):
        smoothed = lowess(series, np.arange(len(series)), frac=frac, return_sorted=False)
        return np.asarray(smoothed).flatten()

    main_smoothed = smooth(main_df['Intensity_polarized'])
    minor_smoothed = smooth(minor_df['Intensity_polarized'])
    full_smoothed = smooth(full_series)

    min_len = min(len(full_smoothed), len(main_smoothed), len(minor_smoothed))
    main_smoothed = main_smoothed[:min_len]
    minor_smoothed = minor_smoothed[:min_len]
    full_smoothed = full_smoothed[:min_len]

    main_list = list(map(float, main_smoothed))
    minor_list = list(map(float, minor_smoothed))
    full_list = list(map(float, full_smoothed))

    dist_main_full, _ = fastdtw(main_list, full_list, dist=lambda x, y: abs(x - y))
    dist_minor_full, _ = fastdtw(minor_list, full_list, dist=lambda x, y: abs(x - y))
    dist_main_minor, _ = fastdtw(main_list, minor_list, dist=lambda x, y: abs(x - y))

    if dist_main_full < dist_minor_full:
        driver = '主角'
    else:
        driver = '次角'

    sync = '同步' if dist_main_minor < 0.8 * max(dist_main_full, dist_minor_full) else '独立'

    plt.figure(figsize=(10, 5))
    plt.plot(full_smoothed, label='全书', color='black', linewidth=2)
    plt.plot(main_smoothed, label='主角', color='blue', linestyle='--')
    plt.plot(minor_smoothed, label='次角', color='red', linestyle='--')
    plt.title(f"{book_id} 情感曲线与DTW距离\n主导角色: {driver} ｜ 同步性: {sync}")
    plt.xlabel("句子位置")
    plt.ylabel("情感强度（Loess平滑）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{book_id}_情感曲线_DTW.png"))
    plt.close()

    results.append({
        '小说编号': book_id,
        '小说类型': novel_type,
        'DTW(全书,主角)': round(dist_main_full, 2),
        'DTW(全书,次角)': round(dist_minor_full, 2),
        'DTW(主角,次角)': round(dist_main_minor, 2),
        '情感节奏主导角色': driver,
        '主-次同步性判断': sync
    })

results_df = pd.DataFrame(results)
results_path = os.path.join(output_folder, "DTW分析结果汇总表.xlsx")
results_df.to_excel(results_path, index=False)
print("分析完成，结果保存于：", results_path)

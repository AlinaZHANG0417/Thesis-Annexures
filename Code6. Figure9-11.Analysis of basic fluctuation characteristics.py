import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

# 1. 参数配置
data_path = r"H:\ZHANGJINGYI-20250330\data\2.句子情感数据\B01-B18句子情感满意原始数据\2.归一化数据"
output_path = r"H:\ZHANGJINGYI-20250330\基础波动特征results"
os.makedirs(output_path, exist_ok=True)

# 2. 定义计算函数（增加均值回归速率）
def calculate_metrics(series):
    metrics = {}
    
    # 波幅
    metrics['Amplitude'] = series.max() - series.min()
    
    # 波动频率（过零点）
    metrics['Frequency'] = len(np.where(np.diff(np.sign(series)))[0])
    
    # 变异系数
    metrics['CV'] = series.std() / abs(series.mean()) if series.mean() != 0 else np.nan

    # 波峰数
    peaks, _ = find_peaks(series)
    metrics['Peaks'] = len(peaks)

    # 波谷数
    troughs, _ = find_peaks(-series)
    metrics['Troughs'] = len(troughs)

    # 均值回归速率：偏离均值后变动的速度
    mean_val = series.mean()
    deviations = np.abs(series - mean_val)
    metrics['MeanReversion'] = np.mean(np.abs(np.diff(deviations)))

    return metrics

# 3. 批量处理18个小说样本
results = []

for i in range(1, 19):
    file_num = str(i).zfill(2)
    filename = f"B{file_num}情感汇总_处理后.csv"
    filepath = os.path.join(data_path, filename)
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        series = df['Intensity_polarized']
        metrics = calculate_metrics(series)
        metrics['Book'] = f'B{file_num}'
        results.append(metrics)
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 4. 保存计算结果
result_df = pd.DataFrame(results).set_index('Book')
result_df.to_csv(os.path.join(output_path, '情感分析统计指标.csv'), encoding='utf-8-sig')

# 5. 绘图：每个指标分别绘制箱线图 + 小提琴图
sns.set(style="whitegrid")
metrics_to_plot = ['Amplitude', 'Frequency', 'CV', 'Peaks', 'Troughs', 'MeanReversion']

for metric in metrics_to_plot:
    # 箱线图
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=result_df[metric], color='skyblue')
    plt.title(f'{metric} 指标箱线图', fontsize=14, fontproperties='SimHei')
    plt.ylabel(metric, fontproperties='SimHei')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'箱线图_{metric}.png'), dpi=300)
    plt.close()

    # 小提琴图
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=result_df[metric], inner='box', color='lightgreen')
    plt.title(f'{metric} 指标小提琴图', fontsize=14, fontproperties='SimHei')
    plt.ylabel(metric, fontproperties='SimHei')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'小提琴图_{metric}.png'), dpi=300)
    plt.close()

print("✅ 所有指标计算与图形绘制完成，已保存至：", output_path)

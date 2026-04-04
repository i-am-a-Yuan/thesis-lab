# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:11:43 2026

@author: 86183
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv("data01.csv")
df = df.rename(columns={"roads_perc":"road1","roads_nodes":"road2","housing_price":"price","pop_density":"pop","pois":"poi","zone_perc":"build"}); 



# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和子图
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# 设置颜色
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 1. 人口密度箱线图
axes[0].boxplot(df['pop'].dropna(), patch_artist=True,
                boxprops=dict(facecolor=colors[0], color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
axes[0].set_title('道路密度箱线图', fontsize=14, fontweight='bold')
axes[0].set_ylabel('道路密度', fontsize=12)
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_xticklabels([''])

# 添加统计信息
pop_stats = df['pop'].describe()
axes[0].text(0.95, 0.95, 
             f"中位数: {pop_stats['50%']:.1f}\n"
             f"均值: {pop_stats['mean']:.1f}\n"
             f"标准差: {pop_stats['std']:.1f}",
             transform=axes[0].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

# 2. 房价箱线图 - 请确认你的房价列名
# 如果列名不是'price'，请修改为正确的列名
try:
    # 假设房价列名为'price'，如果不是请修改
    price_col = 'price'  # 如果列名不同，请修改这里
    axes[1].boxplot(df[price_col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor=colors[1], color='black'),
                    medianprops=dict(color='yellow', linewidth=2),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    axes[1].set_title('房价箱线图', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('房价', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticklabels([''])
    
    # 添加统计信息
    price_stats = df[price_col].describe()
    axes[1].text(0.95, 0.95, 
                 f"中位数: {price_stats['50%']:.1f}\n"
                 f"均值: {price_stats['mean']:.1f}\n"
                 f"标准差: {price_stats['std']:.1f}",
                 transform=axes[1].transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)
except KeyError:
    print(f"警告: 未找到'{price_col}'列。请确认房价列名。")
    # 列出所有列名供参考
    print(f"可用的列名: {list(df.columns)}")

# 3. 道路交叉口数量箱线图
# 假设使用road1作为道路交叉口数量，如果需要使用road2或两者之和，请修改
road_col = 'road1'  # 如果需要使用road2，请修改为'road2'
axes[2].boxplot(df[road_col].dropna(), patch_artist=True,
                boxprops=dict(facecolor=colors[2], color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
axes[2].set_title('道路交叉口数量箱线图', fontsize=14, fontweight='bold')
axes[2].set_ylabel('道路交叉口数量', fontsize=12)
axes[2].grid(True, alpha=0.3, linestyle='--')
axes[2].set_xticklabels([''])

# 添加统计信息
road_stats = df[road_col].describe()
axes[2].text(0.95, 0.95, 
             f"中位数: {road_stats['50%']:.1f}\n"
             f"均值: {road_stats['mean']:.1f}\n"
             f"标准差: {road_stats['std']:.1f}",
             transform=axes[2].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

# 调整布局
plt.tight_layout()

# 添加总标题
fig.suptitle('主要变量箱线图分析：极端值与分布离散程度', 
             fontsize=16, fontweight='bold', y=1.02)

# 显示图形
plt.show()

# 输出极端值信息
print("="*60)
print("极端值检测结果：")
print("="*60)

# 检测极端值（使用IQR方法）
def detect_outliers(series, name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    print(f"\n{name}:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  极端值数量: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  极端值: {outliers.values}")
    return outliers

# 检测各变量的极端值
pop_outliers = detect_outliers(df['pop'], "人口密度")
try:
    price_outliers = detect_outliers(df[price_col], "房价")
except:
    pass
road_outliers = detect_outliers(df[road_col], "道路交叉口数量")

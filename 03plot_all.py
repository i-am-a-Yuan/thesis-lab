# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 06:59:41 2026

@author: 86183 from 02plot.py
"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv("data01.csv")
df = df.rename(columns={"roads_perc":"road1","roads_nodes":"road2","housing_price":"price","pop_density":"pop","pois":"poi","zone_perc":"build"}); 


plt.close('all')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman'] #, 'DejaVu Sans']
# plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'      # 接近 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams

# 创建图形和子图
fig, axes = plt.subplots(2, 3, figsize=(10, 4))

# 设置颜色
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#25A218', '#FB7E21', '#B8A982']
# %% 1. 道路密度箱线图  road1
axes[0][0].boxplot(df['road1'], patch_artist=True,
                boxprops=dict(facecolor=colors[0], color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
axes[0][0].set_title('街镇道路密度', fontsize=14, fontweight='bold')
# axes[0][0].set_ylabel('%', fontsize=12, rotation=0) # 横向文本
axes[0][0].set_ylabel(r'$\mathit{\%}$', fontsize=12, rotation=0) # 横向文本

axes[0][0].yaxis.set_label_coords(-0.05, 1.02) # y轴标签置于y轴正上方
axes[0][0].grid(True, alpha=0.3, linestyle='--')
axes[0][0].set_xticklabels([''])

# 添加统计信息
pop_stats = df['road1'].describe()
axes[0][0].text(0.95, 0.95, 
             f"中位数: {pop_stats['50%']:.3f}\n"
             f"均值: {pop_stats['mean']:.3f}\n"
             f"标准差: {pop_stats['std']:.3f}",
             transform=axes[0][0].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)
# %% 2. 道路交叉口数量箱线图  road2
axes[0][1].boxplot(df['road2'], patch_artist=True,
                boxprops=dict(facecolor=colors[1], color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
axes[0][1].set_title('街镇道路交叉口数量', fontsize=14, fontweight='bold')
axes[0][1].set_ylabel('个', fontsize=12, rotation=0) # 横向文本
axes[0][1].yaxis.set_label_coords(-0.05, 1.02) # y轴标签置于y轴正上方
axes[0][1].grid(True, alpha=0.3, linestyle='--')
axes[0][1].set_xticklabels([''])

# 添加统计信息
pop_stats = df['road2'].describe()
axes[0][1].text(0.95, 0.95, 
             f"中位数: {pop_stats['50%']:.1f}\n"
             f"均值: {pop_stats['mean']:.1f}\n"
             f"标准差: {pop_stats['std']:.1f}",
             transform=axes[0][1].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

# %% 3. gdp箱线图  gdp
axes[0][2].boxplot(df['gdp'], patch_artist=True,
                boxprops=dict(facecolor=colors[2], color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
axes[0][2].set_title('街镇GDP总量', fontsize=14, fontweight='bold')
axes[0][2].set_ylabel('亿元', fontsize=12, rotation=0) # 横向文本
axes[0][2].yaxis.set_label_coords(-0.05, 1.02) # y轴标签置于y轴正上方
axes[0][2].grid(True, alpha=0.3, linestyle='--')
axes[0][2].set_xticklabels([''])

# 添加统计信息
pop_stats = df['gdp'].describe()
axes[0][2].text(0.95, 0.95, 
             f"中位数: {pop_stats['50%']:.1f}\n"
             f"均值: {pop_stats['mean']:.1f}\n"
             f"标准差: {pop_stats['std']:.1f}",
             transform=axes[0][2].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

# %% 4. price箱线图  price
axes[1][0].boxplot(df['price'], patch_artist=True,
                boxprops=dict(facecolor=colors[3], color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
axes[1][0].set_title('街镇单位面积平均房价', fontsize=14, fontweight='bold')
# axes[1][0].set_ylabel('元/m$^2$', fontsize=12, rotation=0) # 横向文本
# axes[1][0].set_ylabel(r'$\mathit{yuan/m^2}$', fontsize=12, rotation=0) # 横向文本
axes[1][0].set_ylabel('元/' + r'$\mathit{m^2}$', fontsize=12, rotation=0) # 横向文本

axes[1][0].yaxis.set_label_coords(-0.05, 1.02) # y轴标签置于y轴正上方
axes[1][0].grid(True, alpha=0.3, linestyle='--')
axes[1][0].set_xticklabels([''])

# 添加统计信息
pop_stats = df['price'].describe()
axes[1][0].text(0.95, 0.95, 
             f"中位数: {pop_stats['50%']:.1f}\n"
             f"均值: {pop_stats['mean']:.1f}\n"
             f"标准差: {pop_stats['std']:.1f}",
             transform=axes[1][0].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

# %% 5. poi箱线图  poi
axes[1][1].boxplot(df['poi'], patch_artist=True,
                boxprops=dict(facecolor=colors[4], color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
axes[1][1].set_title('街镇POI数量', fontsize=14, fontweight='bold')
axes[1][1].set_ylabel('个', fontsize=12, rotation=0) # 横向文本
axes[1][1].yaxis.set_label_coords(-0.05, 1.02) # y轴标签置于y轴正上方
axes[1][1].grid(True, alpha=0.3, linestyle='--')
axes[1][1].set_xticklabels([''])

# 添加统计信息
pop_stats = df['poi'].describe()
axes[1][1].text(0.95, 0.95, 
             f"中位数: {pop_stats['50%']:.1f}\n"
             f"均值: {pop_stats['mean']:.1f}\n"
             f"标准差: {pop_stats['std']:.1f}",
             transform=axes[1][1].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

# %% 6. build箱线图  build
axes[1][2].boxplot(df['build'], patch_artist=True,
                boxprops=dict(facecolor=colors[5], color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
axes[1][2].set_title('街镇建设用地占比', fontsize=14, fontweight='bold')
# axes[1][2].set_ylabel('%', fontsize=12, rotation=0) # 横向文本
axes[1][2].set_ylabel(r'$\mathit{\%}$', fontsize=12, rotation=0) # 横向文本

axes[1][2].yaxis.set_label_coords(-0.05, 1.02) # y轴标签置于y轴正上方
axes[1][2].grid(True, alpha=0.3, linestyle='--')
axes[1][2].set_xticklabels([''])

# 添加统计信息
pop_stats = df['gdp'].describe()
axes[1][2].text(0.95, 0.95, 
             f"中位数: {pop_stats['50%']:.1f}\n"
             f"均值: {pop_stats['mean']:.1f}\n"
             f"标准差: {pop_stats['std']:.1f}",
             transform=axes[1][2].transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)




# %%


# 调整布局
plt.tight_layout()

# 添加总标题
# fig.suptitle('主要变量箱线图分析：极端值与分布离散程度', 
#              fontsize=16, fontweight='bold', y=1.02)

# fig.suptitle('上海公园绿地可达性主要自变量箱线图', 
#              fontsize=16, fontweight='bold', y=1.02)


# 显示图形
plt.show()



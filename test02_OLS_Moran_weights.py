# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 16:07:10 2026

@author: 86183
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm

# from libpysal.weights import Queen
from libpysal.weights import Queen, DistanceBand, KNN
from esda.moran import Moran
import matplotlib.pyplot as plt


# 读取shp
gdf = gpd.read_file("boundary.shp")

## 这里的 df2 是去除 ln_pop 建立的
df2 = pd.read_csv("df2.csv")


# 合并模型数据（假设按 STREET 字段合并）
gdf = gdf.merge(df2, on="STREET")


# 使用 Queen 邻接矩阵
w_queen = Queen.from_dataframe(gdf)
w_queen.transform = "r"   # 行标准化

# Distance
w_Distance = DistanceBand.from_dataframe(gdf, threshold=2000)
w_Distance.transform = "r"

# KNN
w_K2 = KNN.from_dataframe(gdf, k=2); w_K2.transform = "r"
w_K3 = KNN.from_dataframe(gdf, k=3); w_K3.transform = "r"
w_K4 = KNN.from_dataframe(gdf, k=4); w_K4.transform = "r"
w_K5 = KNN.from_dataframe(gdf, k=5); w_K5.transform = "r"
w_K6 = KNN.from_dataframe(gdf, k=6); w_K6.transform = "r"
w_K7 = KNN.from_dataframe(gdf, k=7); w_K7.transform = "r"
w_K8 = KNN.from_dataframe(gdf, k=8); w_K8.transform = "r"

w = w_queen

# 全局 Moran's I 检验（对 ln_access）
y = gdf["ln_access"].values

moran = Moran(y, w)

print("Moran's I:", moran.I)
print("p-value:", moran.p_sim)

### 残差 Moran's I 检验

# X = gdf[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build" ]]
X2 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build" ]] # - ln_pop
X4 = df2[[ "road1", "ln_road2", "ln_price", "build" ]] # - ln _gdp


X = X4
X = sm.add_constant(X)
y = gdf["ln_access"]


model2 = sm.OLS(y, X).fit()

residuals = model2.resid.values

# 2. 两个检验对象
targets = {
    "ln_access": gdf["ln_access"].values,
    "residual": residuals
}

##### 多个权重矩阵下的检验
weights = {
    "Queen": w_queen,
    "Distance": w_Distance,
    "K2": w_K2,
    "K3": w_K3,
    "K4": w_K4,
    "K5": w_K5,
    "K6": w_K6,
    "K7": w_K7,
    "K8": w_K8
}

# 4. 生成表格
table = pd.DataFrame(index=["ln_access", "residual"], columns=weights.keys())

for row_name, values in targets.items():
    for col_name, w in weights.items():
        moran = Moran(values, w)
        table.loc[row_name, col_name] = f"{moran.I:.3f}\n[{moran.p_norm:.3f}]"

print(table)

table.to_excel("OLS2_Moran_lnaccess_residual.xlsx")

# table.to_excel("OLS4_Moran_lnaccess_residual.xlsx")
OLS4_Moran_lnaccess_residual = table


# moran_resid = Moran(residuals, w)


#### LM 检验代码

from spreg import OLS
from libpysal.weights import Queen

# 权重矩阵已存在：w

y_G = gdf["ln_access"].values.reshape(-1,1)
X_G = gdf[["road1","ln_road2","ln_gdp","ln_price","ln_poi","build"]].values

w = w_Distance
w = w_K3
w = w_K4
w = w_K5
w = w_K6
w = w_K3
w = w_K3
w = w_K3





ols_sp = OLS(y_G, X_G, w=w, spat_diag=True, name_y='ln_access')

# print("LM-Lag:", ols_sp.lm_lag)
# print("LM-Lag p-value:", ols_sp.lm_lag[1])

# print("LM-Error:", ols_sp.lm_error)
# print("LM-Error p-value:", ols_sp.lm_error[1])

# print("Robust LM-Lag:", ols_sp.rlm_lag)
# print("Robust LM-Error:", ols_sp.rlm_error)

lm_result = pd.DataFrame({
    "Statistic":[
        ols_sp.lm_lag[0],
        ols_sp.rlm_lag[0],
        ols_sp.lm_error[0],
        ols_sp.rlm_error[0]
    ],
    "p_value":[
        ols_sp.lm_lag[1],
        ols_sp.rlm_lag[1],
        ols_sp.lm_error[1],
        ols_sp.rlm_error[1]
    ]
},
index=["LM_Lag","Robust_LM_Lag","LM_Error","Robust_LM_Error"])

lm_result

# lm_result.to_excel("LM_test.xlsx")




















# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:24:58 2026

@author: 86183
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm

import spreg
from libpysal.weights import Queen, DistanceBand, KNN

# 读取shp
gdf = gpd.read_file("boundary.shp")

## 这里的 df2 是去除 ln_pop 建立的
df2 = pd.read_csv("df2.csv")


# 合并模型数据（假设按 STREET 字段合并）
gdf = gdf.merge(df2, on="STREET")
# gdf.to_excel("gdf.xlsx")

# 因变量
# y = gdf["ln_access"].values#.reshape(-1,1)


# 自变量
# X = gdf[["road1","road2","gdp","price","pop","poi","build"]].values # 所有原自变量
# X = gdf[["road1","road2","gdp","price","poi","build"]].values # 去除pop所有原自变量
# X = gdf[["road1", "ln_road2", "ln_gdp", "ln_price", "ln_pop", "ln_poi", "build"]] # 所有对数自变量 OLS 1
# X = gdf[["road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build"]] # 去除ln_pop 的对数自变量

# X = gdf[["ln_pop"]] # 只有ln_pop 的对数自变量
X2 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build" ]] # - ln_pop
X4 = df2[[ "road1", "ln_road2", "ln_price", "build" ]] # - ln _gdp


X = X4
X = sm.add_constant(X)
y = gdf["ln_access"]



# 构建权重矩阵
w_Queen = Queen.from_dataframe(gdf); w_Queen.transform = "r"
w_Distance = DistanceBand.from_dataframe(gdf, threshold=6000); w_Distance.transform = "r"
w_K2 = KNN.from_dataframe(gdf, k=2); w_K2.transform = "r"
w_K3 = KNN.from_dataframe(gdf, k=3); w_K3.transform = "r"
w_K4 = KNN.from_dataframe(gdf, k=4); w_K4.transform = "r"
w_K5 = KNN.from_dataframe(gdf, k=5); w_K5.transform = "r"
w_K6 = KNN.from_dataframe(gdf, k=6); w_K6.transform = "r"
w_K7 = KNN.from_dataframe(gdf, k=7); w_K7.transform = "r"
w_K8 = KNN.from_dataframe(gdf, k=8); w_K8.transform = "r"

weights = {
    "Queen": w_Queen,
    "Distance": w_Distance,
    "K2": w_K2,
    "K3": w_K3,
    "K4": w_K4,
    "K5": w_K5,
    "K6": w_K6,
    "K7": w_K7,
    "K8": w_K8
}


### 
rows = ["LM_Lag", "Robust_LM_Lag", "LM_Error", "Robust_LM_Error"]
cols = pd.MultiIndex.from_product(
    [weights.keys(), ["Statistic", "p_value"]],
    names=["Weight", "Type"]
)

table = pd.DataFrame(index=rows, columns=cols)

for name, w in weights.items():
    model = spreg.OLS(y, X, w=w, spat_diag=True)
    table.loc["LM_Lag", (name, "Statistic")] = model.lm_lag[0]
    table.loc["LM_Lag", (name, "p_value")] = model.lm_lag[1]
    table.loc["Robust_LM_Lag", (name, "Statistic")] = model.rlm_lag[0]
    table.loc["Robust_LM_Lag", (name, "p_value")] = model.rlm_lag[1]
    table.loc["LM_Error", (name, "Statistic")] = model.lm_error[0]
    table.loc["LM_Error", (name, "p_value")] = model.lm_error[1]
    table.loc["Robust_LM_Error", (name, "Statistic")] = model.rlm_error[0]
    table.loc["Robust_LM_Error", (name, "p_value")] = model.rlm_error[1]

table = table.astype(float).round(3)
# table.to_excel("LM_tests.xlsx")  # 对所有原自变量进行LM检验
# table.to_excel("LM_tests_2.xlsx")  # 对除pop的所有自变量进行LM检验
# table.to_excel("LM_tests_ln_1.xlsx")
# table.to_excel("LM_tests_ln_2.xlsx") # 对除ln_pop的对数自变量进行LM检验
table.to_excel("LM_tests_ln_pop.xlsx")  # 只有ln_pop的自变量进行LM检验
print(table)
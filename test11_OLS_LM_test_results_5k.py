# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:59:34 2026

@author: 86183
"""

import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
import spreg
from libpysal.weights import Queen, DistanceBand, KNN


df2 = pd.read_csv("df2_5k.csv")

gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")

y = df2["ln_access"].values.reshape(-1,1)

# X2 = df2[["road1","ln_road2","ln_gdp","ln_price","ln_poi","build"]]
X2 = df2[["road1","ln_road2","ln_gdp","ln_price","ln_poi","build"]]

y = df2["ln_access"]

X = X2

# 权重矩阵
weights = {
    "Queen": Queen.from_dataframe(gdf),
    "Distance": DistanceBand.from_dataframe(gdf, threshold=6000),
    "K2": KNN.from_dataframe(gdf, k=2),
    "K3": KNN.from_dataframe(gdf, k=3),
    "K4": KNN.from_dataframe(gdf, k=4),
    "K5": KNN.from_dataframe(gdf, k=5),
    "K6": KNN.from_dataframe(gdf, k=6),
    "K7": KNN.from_dataframe(gdf, k=7),
    "K8": KNN.from_dataframe(gdf, k=8)
}

for w in weights.values():
    w.transform = "r"

rows = ["LM_Lag","Robust_LM_Lag","LM_Error","Robust_LM_Error"]
table = pd.DataFrame(index=rows, columns=weights.keys())

for name, w in weights.items():
    model = spreg.OLS(y, X, w=w, spat_diag=True)
    
    # table.loc["LM_Lag", name] = f"{model.lm_lag[0]:.3f} ({model.lm_lag[1]:.3f})"
    # table.loc["Robust_LM_Lag", name] = f"{model.rlm_lag[0]:.3f} ({model.rlm_lag[1]:.3f})"
    # table.loc["LM_Error", name] = f"{model.lm_error[0]:.3f} ({model.lm_error[1]:.3f})"
    # table.loc["Robust_LM_Error", name] = f"{model.rlm_error[0]:.3f} ({model.rlm_error[1]:.3f})"
    
    table.loc["LM_Lag", name] = f"{model.lm_lag[0]:.3f} [{model.lm_lag[1]:.3f}]"
    table.loc["Robust_LM_Lag", name] = f"{model.rlm_lag[0]:.3f} [{model.rlm_lag[1]:.3f}]"
    table.loc["LM_Error", name] = f"{model.lm_error[0]:.3f} [{model.lm_error[1]:.3f}]"
    table.loc["Robust_LM_Error", name] = f"{model.rlm_error[0]:.3f} [{model.rlm_error[1]:.3f}]"

table.to_excel("OLS_LM_test_results_5k.xlsx")
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:57:52 2026

@author: 86183
"""
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
import spreg
from libpysal.weights import Queen, DistanceBand, KNN


df2 = pd.read_csv("df2.csv")

# 读取shp
gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")

X2 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build" ]] # - ln_pop
X4 = df2[[ "road1", "ln_road2", "ln_price", "build" ]] # - ln _gdp

X2 = sm.add_constant(X2)
X4 = sm.add_constant(X4)


raw_X1 = df2[["road1", "road2", "gdp", "price", "pop", "poi", "build"]]
raw_X2 = df2[["road1", "road2", "gdp", "price", "poi", "build"]]
raw_X3 = df2[["road1", "road2", "gdp", "price", "pop", "poi", "build"]]

raw_X1 = sm.add_constant(raw_X1)
raw_X2 = sm.add_constant(raw_X2)


y = df2["ln_access"]

X = X4

# 权重矩阵
weights = {
    "Queen": Queen.from_dataframe(gdf),
    "Distance": DistanceBand.from_dataframe(gdf, threshold=2000),
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


# OLS2_LM_test_weights = table
# table.to_excel("OLS2_LM_test_results.xlsx")

# OLS4_LM_test_weights = table
# table.to_excel("OLS4_LM_test_results.xlsx")

# raw_OLS4_LM_test_weights = table
# table.to_excel("raw_OLS1_LM_test_results.xlsx")

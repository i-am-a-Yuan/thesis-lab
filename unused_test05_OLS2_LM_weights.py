# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:57:42 2026

@author: 86183
"""
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import Queen, DistanceBand, KNN

df2 = pd.read_csv("df2.csv")

# 读取shp
gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")



# 1. OLS2
# X = df[["road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build"]]
X2 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build" ]] # - ln_pop
X4 = df2[[ "road1", "ln_road2", "ln_price", "build" ]] # - ln _gdp

X2 = sm.add_constant(X2)
X4 = sm.add_constant(X4)

y = df2["ln_access"]

model_OLS2 = sm.OLS(y, X4).fit()
resid = model_OLS2.resid.values

# 2. 权重矩阵
w_Queen = Queen.from_dataframe(gdf); w_Queen.transform = "r"
w_Distance = DistanceBand.from_dataframe(gdf, threshold=2000); w_Distance.transform = "r"
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

# 3. Moran's I 检验表
index = ["ln_access", "residual"]
table = pd.DataFrame(index=index, columns=weights.keys())

for name, w in weights.items():
    moran = Moran(resid, w)
    table.loc["Moran_I", name] = f"{moran.I:.3f}\n[{moran.p_norm:.3f}]"
    # table.loc["E_I", name]     = f"{moran.EI:.3f}[-]"
    # table.loc["z_norm", name]  = f"{moran.z_norm:.3f}[{moran.p_norm:.3f}]"
    # table.loc["p_norm", name]  = f"{moran.p_norm:.3f}[{moran.p_norm:.3f}]"

# print(table)
# table.to_excel("OLS2_Moran_lnaccess_residual.xlsx")
table.to_excel("OLS4_Moran_lnaccess_residual.xlsx")

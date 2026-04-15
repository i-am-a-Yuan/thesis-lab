# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 05:40:12 2026

@author: 86183
"""

# %%

import numpy as np
import pandas as pd
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import Queen, DistanceBand, KNN
from spreg import ML_Lag, ML_Error, OLS
from scipy.stats import chi2

# ---------------------------
# 数据
# ---------------------------
df2 = pd.read_csv("df2_5k.csv")
df2["ln_access"] = np.log(df2["access"])
df2["delta"] = df2["ln_access_5k"] - df2["ln_access"]

gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")

# y = df2["ln_access_5k"].values #.reshape(-1,1)
y = df2["delta"]

X2 = df2[["road1","ln_road2","ln_gdp","ln_price","ln_poi","build"]]
# X4 = df2[[ "road1", "ln_road2", "ln_price", "build" ]] # - ln _gdp

X = X2
# X = X4
# ---------------------------
# 显著性函数
# ---------------------------
def star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

# ---------------------------
# 权重矩阵
# ---------------------------
w_Queen = Queen.from_dataframe(gdf); w_Queen.transform = "r"

weights = {
    "Queen": w_Queen,
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

# ---------------------------
# 行名
# ---------------------------
param_rows = [
    "CONSTANT",
    "road1","ln_road2","ln_gdp","ln_price","ln_poi","build",
    "ln_access",
    "W_road1","W_ln_road2","W_ln_gdp","W_ln_price","W_ln_poi","W_build",
    # "rho"
    "W_ln_access"
]

stat_rows = [
    "Residual Moran's I",
    "R2",
    "Log Likelihood",
    "Akaike Info Criterion",
    "Schwarz Criterion",
    "Wald",
    "Wald_p",
    # "Likelihood Ratio (LR)",
    "LR_SAR",
    "LR_SAR_p",
    # "LR_p",
    "LR_SEM",
    "LR_SEM_p",
    "Lagrange Multiplier (LM)"
]

rows = param_rows + stat_rows
table = pd.DataFrame(index=rows, columns=weights.keys())

# ---------------------------
# 循环估计 SDM
# ---------------------------
for name, w in weights.items():

    # OLS 基准
    ols = OLS(y, X.values, w=w, spat_diag=True)

    # SAR 基准
    sar = ML_Lag(
        y,
        X.values,
        w=w,
        slx_lags=0,
        name_y="ln_access",
        name_x=list(X.columns)
    )
    
    # SEM 基准
    sem = ML_Error(
        y,
        X.values,
        w=w,
        name_y="ln_access",
        name_x=list(X.columns)
    )


    # SDM (Durbin)
    sdm = ML_Lag(
        y,
        X.values,
        w=w,
        slx_lags=1,
        name_y="ln_access",
        name_x=list(X.columns),

    )

    betas = np.array(sdm.betas).flatten()

    std = np.array(sdm.std_err).flatten()

    pvals = [i[1] for i in sdm.z_stat]

    names = sdm.name_x

    for n,c,se,p in zip(names,betas,std,pvals):
        table.loc[n,name] = f"{c:.3f}{star(p)}\n({se:.3f})"
    
    # table.loc["rho", name] = f"{sdm.rho:.3f{star()}}

    # ------------------
    # 残差 Moran's I
    # ------------------
    moran = Moran(sdm.u, w)
    table.loc["Residual Moran's I",name] = f"{moran.I:.3f}\n[{moran.p_norm:.3f}]"

    # ------------------
    # R2
    # ------------------
    if hasattr(sdm,"pr2"):
        table.loc["R2",name] = f"{sdm.pr2:.3f}"

    # ------------------
    # LogLik / AIC / SC
    # ------------------
    table.loc["Log Likelihood",name] = f"{sdm.logll:.3f}"
    table.loc["Akaike Info Criterion",name] = f"{sdm.aic:.3f}"
    table.loc["Schwarz Criterion",name] = f"{sdm.schwarz:.3f}"


    k = X.shape[1]
    # ------------------
    # Wald
    # ------------------
    rho = betas[-1]
    se = std[-1]
    # wald = (rho/se)**2  # 对比OLS
    wx_coefs = betas[1+k:1+2*k]
    wx_se = std[1+k:1+2*k]
    wald = sum((wx_coefs / wx_se)**2)
    
    wald_p_value = 1 - chi2.cdf(wald, df=k)
    table.loc["Wald",name] = f"{wald:.3f}"
    table.loc["Wald_p", name] = f"{wald_p_value:.3f}"

    # ------------------
    # LR
    # ------------------
    # lr = 2*(sdm.logll - ols.logll)
    # LR 比较 SAR
    lr_sar = 2 * (sdm.logll - sar.logll)
    lr_sar_p = 1 - chi2.cdf(lr_sar, df=k)

    table.loc["LR_SAR",name] = f"{lr_sar:.3f}"
    table.loc["LR_SAR_p", name] = f"{lr_sar_p:.3f}"

    lr_sem = 2 * (sdm.logll - sem.logll)
    lr_sem_p = 1 - chi2.cdf(lr_sem, df=k)
    
    table.loc["LR_SEM",name] = f"{lr_sem:.3f}"
    table.loc["LR_SEM_p", name] = f"{lr_sem_p:.3f}"

    # ------------------
    # LM
    # ------------------
    table.loc["Lagrange Multiplier (LM)",name] = f"{ols.lm_lag[0]:.3f}"
    
# ---------------------------
# 输出
# ---------------------------
# table.to_excel("SDM1_results_multiW_delta.xlsx")
print(table)



# %%
1

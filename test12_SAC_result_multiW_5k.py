# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:19:17 2026

@author: 86183
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.api as sm
# import spreg
from libpysal.weights import Queen, DistanceBand, KNN
from spreg import ML_Lag, ML_Error, OLS
from spreg import GM_Combo
# from spreg import ML_Combo
from spreg import diagnostics
from esda.moran import Moran
from scipy.stats import chi2



df2 = pd.read_csv("df2_5k.csv")

gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")

y = df2["ln_access_5k"].values.reshape(-1,1)

X2 = df2[["road1","ln_road2","ln_gdp","ln_price","ln_poi","build"]]

# y = df2["ln_access"]

X = X2

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
# %%


# ---------------------------
# 行名
# ---------------------------
param_rows = [
    "CONSTANT",
    "road1","ln_road2","ln_gdp","ln_price","ln_poi","build",
    # "W_road1","W_ln_road2","W_ln_gdp","W_ln_price","W_ln_poi","W_build",
    # "rho"
    "W_ln_access",
    "lambda"
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
# %%


rows = param_rows + stat_rows
table = pd.DataFrame(index=rows, columns=weights.keys())

# ---------------------------
# 循环估计 SDM
# ---------------------------
for name, w in weights.items():

    # OLS 基准
    ols = OLS(y, X.values, w=w, spat_diag=True)

    # SAC（Spatial Autoregressive Combined / SARAR 模型）
    sac = GM_Combo(
        y,
        X.values,
        w=w,
        name_y="ln_access",
        name_x=list(X.columns)
    )
    # sac = ML_Combo(
    #     y,
    #     X.values,
    #     w=w,
    #     name_y="ln_access",
    #     name_x=list(X.columns)
    # )
    
    
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
    # sdm = ML_Lag(
    #     y,
    #     X.values,
    #     w=w,
    #     slx_lags=1,
    #     name_y="ln_access",
    #     name_x=list(X.columns)
    # )

    betas = np.array(sac.betas).flatten()

    # std = np.array(sac.std_err).flatten()
    std = np.array(sac.std_err).flatten()
    std = np.append(std, sac.std_y)


    pvals = [i[1] for i in sac.z_stat]

    # names = sac.name_x + sac.name_yend
    names = sac.name_z


    for n,c,se,p in zip(names,betas,std,pvals):
        table.loc[n,name] = f"{c:.3f}{star(p)}\n({se:.3f})"
    
    # table.loc["lambda"]
    n_x = len(sac.name_x)
    p = sac.z_stat[n_x][1]
    se = std[n_x]
    table.loc["W_ln_access_5k", name] = f"{sac.rho[0]:.3f}{star(p)}\n({se:.3f})"
    
    table.loc["lambda", name] = f"{betas[n_x + 1]:.3f}"

    # ------------------
    # 残差 Moran's I
    # ------------------
    moran = Moran(sac.u, w)
    table.loc["Residual Moran's I",name] = f"{moran.I:.3f}\n[{moran.p_norm:.3f}]"

    # ------------------
    # R2
    # ------------------
    if hasattr(sac,"pr2"):
        table.loc["R2",name] = f"{sac.pr2:.3f}"

    # ------------------
    # LogLik / AIC / SC
    # ------------------
    sac.utu = np.sum((y - sac.predy)**2)
    log_ll = diagnostics.log_likelihood(sac)
    sac.logll = log_ll
    table.loc["Log Likelihood",name] = f"{log_ll:.3f}" 
    aic = diagnostics.akaike(sac)
    table.loc["Akaike Info Criterion",name] = f"{aic:.3f}"
    schwarz = diagnostics.schwarz(sac)
    table.loc["Schwarz Criterion",name] = f"{schwarz:.3f}"


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
    lr = 2*(sac.logll - ols.logll)
    # LR 比较 SAR
    lr_sar = 2 * (sac.logll - sar.logll)
    lr_sar_p = 1 - chi2.cdf(lr_sar, df=k)

    table.loc["LR_SAR",name] = f"{lr_sar:.3f}"
    table.loc["LR_SAR_p", name] = f"{lr_sar_p:.3f}"

    lr_sem = 2 * (sac.logll - sem.logll)
    lr_sem_p = 1 - chi2.cdf(lr_sem, df=k)
    
    table.loc["LR_SEM",name] = f"{lr_sem:.3f}"
    table.loc["LR_SEM_p", name] = f"{lr_sem_p:.3f}"

    # # ------------------
    # # LM
    # # ------------------
    table.loc["Lagrange Multiplier (LM)",name] = f"{ols.lm_lag[0]:.3f}"
    
# ---------------------------
# 输出
# ---------------------------
# table.to_excel("SAC_result_multiW_5k.xlsx")
# 







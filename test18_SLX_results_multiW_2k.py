# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:24:37 2026

@author: 86183
"""


import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
import spreg
from libpysal.weights import Queen, DistanceBand, KNN


df2 = pd.read_csv("df2.csv", )

# 读取shp
gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")

X2 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build" ]] # - ln_pop
# X4 = df2[[ "road1", "ln_road2", "ln_price", "build" ]] # - ln _gdp
X2 = sm.add_constant(X2)
# X4 = sm.add_constant(X4)

y = df2["ln_access"]

#### switch SEM1 or SEM2
X = X2
# X = X4

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

############################################

# %%

import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from spreg import OLS, ML_Error

def star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

# 因变量、自变量（spreg 需要 numpy）
# y_arr = y.values.reshape(-1, 1)
y_arr = y.values
X_arr = X.values

# 行名：常数项 + 各解释变量 + lambda + 统计量
# param_rows = list(X.columns) + ["lambda"]

# ---------------------------
# 行名
# ---------------------------
param_rows = [
    "CONSTANT",
    "road1","ln_road2","ln_gdp","ln_price","ln_poi","build",
    "W_road1","W_ln_road2","W_ln_gdp","W_ln_price","W_ln_poi","W_build",
    # "rho"
    # "W_ln_access"
    # "lambda"
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
    # "LR_SAR",
    # "LR_SAR_p",
    # "LR_p",
    "LR_SEM",
    "LR_SEM_p",
    "Lagrange Multiplier (LM)"
]

stat_rows = [
    "Residual Moran's I",
    "R2",
    "Log Likelihood",
    "Akaike Info Criterion",
    "Schwarz Criterion",
    "Wald",
    "Likelihood Ratio (LR)",
    "Lagrange Multiplier (LM)"
]
rows = param_rows + stat_rows

# 结果表
slx_table = pd.DataFrame(index=rows, columns=list(weights.keys()))

for w_name, w in weights.items():
    # w_sparse = w.sparse
    # 1. OLS（用于 LM_Error）
    ols_model = OLS(y_arr, X_arr, w=w, spat_diag=True, name_y="ln_access", name_x=list(X.columns))

    # 2. SEM
    # slx_model = ML_Error(y_arr, X_arr, w=w, name_y="ln_access", name_x=list(X.columns))
    # slx_model = ML_Error(y_arr, X_arr, w=w_sparse)
    slx_model = OLS(
        y,
        X,
        w=w,
        slx_lags = 1,
        name_y="ln_access",
        name_x=list(X.columns)
        # name_x=["const","road1","ln_road2","ln_gdp","ln_price","ln_poi","build"] # for SEM1
        # name_x=["const","road1","ln_road2","ln_price","build"] # for SEM2
    )
    
    

    # ---------- 系数、标准差、显著性 ----------
    # betas 通常包含：const、各解释变量、lambda
    betas = np.asarray(slx_model.betas).flatten()

    # 优先从 vm1 提取标准差（通常包含 lambda）；否则退回 vm/std_err
    # if hasattr(slx_model, "vm1") and slx_model.vm1 is not None: # 注释掉 因为只有2个值
    #     se_all = np.sqrt(np.diag(np.asarray(slx_model.vm1)))
    #     print(se_all)
    # if hasattr(slx_model, "vm") and slx_model.vm is not None:
    #     se_all = np.sqrt(np.diag(np.asarray(slx_model.vm)))
    #     print(se_all)
    if hasattr(slx_model, "std_err"):
        se_all = np.asarray(slx_model.std_err).flatten()
        # print(se_all)
    # else:
        # se_all = np.full(len(betas), np.nan)
    # print("se_all", se_all)
    # z_stat: [(z, p), ...]
    t_stats = slx_model.t_stat if hasattr(slx_model, "t_stat") else [(np.nan, np.nan)] * len(betas)

    for i, row_name in enumerate(param_rows):
        # print("i", i)
        coef = betas[i] if i < len(betas) else np.nan
        se = se_all[i] if i < len(se_all) else np.nan
        p = t_stats[i][1] if i < len(t_stats) else np.nan
        slx_table.loc[row_name, w_name] = f"{coef:.3f}{star(p)}\n({se:.3f})"
    
    if w_name == 'Distance':
        print(1)
        
    
    # ---------- 残差 Moran's I ----------
    resid = np.asarray(slx_model.u).flatten()
    moran_resid = Moran(resid, w)
    slx_table.loc["Residual Moran's I", w_name] = f"{moran_resid.I:.3f}\n[{moran_resid.p_norm:.3f}]"

    # ---------- R2 ----------
    # SEM 通常没有传统 OLS R2，这里优先用 pr2（pseudo R2）
    # r2_value = getattr(slx_model, "pr2", np.nan)
    # slx_table.loc["R2", w_name] = f"{r2_value:.3f}"
    # 对于SLX来说 R2 和 OLS 含义是相同的
    r2 = getattr(slx_model, "r2", np.nan)
    slx_table.loc["R2", w_name] = f"{r2:.3f}"
    

    # ---------- 信息准则 ----------
    logll = getattr(slx_model, "logll", np.nan)
    aic = getattr(slx_model, "aic", np.nan)
    schwarz = getattr(slx_model, "schwarz", np.nan)

    slx_table.loc["Log Likelihood", w_name] = f"{logll:.3f}"
    slx_table.loc["Akaike Info Criterion", w_name] = f"{aic:.3f}"
    print(f"{aic:.3f}")
    slx_table.loc["Schwarz Criterion", w_name] = f"{schwarz:.3f}"

    # ---------- Wald 统计量 ----------
    # 对 lambda 的 Wald 检验：W = (lambda / se_lambda)^2
    # lambda_idx = len(param_rows) - 1
    # lam = betas[lambda_idx] if lambda_idx < len(betas) else np.nan
    # se_lam = se_all[lambda_idx] if lambda_idx < len(se_all) else np.nan
    # wald = (lam / se_lam) ** 2 if pd.notna(lam) and pd.notna(se_lam) and se_lam != 0 else np.nan
    # slx_table.loc["Wald", w_name] = f"{wald:.3f}"

    # ---------- LR 统计量 ----------
    # 与同一权重矩阵下 OLS 比较：LR = 2 * (LL_sem - LL_ols)
    # OLS 的 logll 手工计算
    n = len(y_arr)
    rss = float(np.sum(np.asarray(ols_model.u).flatten() ** 2))
    ll_ols = -n / 2 * (np.log(2 * np.pi) + 1 + np.log(rss / n))
    lr = 2 * (logll - ll_ols) if pd.notna(logll) else np.nan
    slx_table.loc["Likelihood Ratio (LR)", w_name] = f"{lr:.3f}"

    # ---------- LM 统计量 ----------
    # 采用同一权重矩阵下 OLS 的 LM_Error
    lm = ols_model.lm_error[0] if hasattr(ols_model, "lm_error") else np.nan
    slx_table.loc["Lagrange Multiplier (LM)", w_name] = f"{lm:.3f}"

# 导出
# slx_table.to_excel("SLX1_multiW_2k.xlsx")

# print(slx_table)


# %%



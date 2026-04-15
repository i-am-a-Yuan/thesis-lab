# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:53:17 2026

@author: 86183
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import Queen, DistanceBand, KNN
from spreg import ML_Lag, ML_Error, OLS
import statsmodels.api as sm
from scipy.stats import chi2


# ---------------------------
# 数据
# ---------------------------
df2 = pd.read_csv("df2_5k.csv")

gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")

y = df2["ln_access_5k"]#.values #.reshape(-1,1)

X2 = df2[["road1","ln_road2","ln_gdp","ln_price","ln_poi","build"]]
# X2 = df2[["road1","ln_road2",         "ln_price","ln_poi","build"]]
# X2 = df2[[        "ln_road2",         "ln_price","ln_poi","build"]]

# X4 = df2[[ "road1", "ln_road2", "ln_price", "build" ]] # - ln _gdp

X = X2

X = sm.add_constant(X)
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
    
y_arr = y.values
X_arr = X.values

# ---------------------------
# 行名
# ---------------------------
param_rows = [
    "CONSTANT",
    "road1","ln_road2","ln_gdp","ln_price","ln_poi","build",
    # "W_road1","W_ln_road2","W_ln_gdp","W_ln_price","W_ln_poi","W_build",
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
    "Likelihood Ratio (LR)",
    # "LR_SAR",
    # "LR_SAR_p",
    # "LR_p",
    # "LR_SEM",
    # "LR_SEM_p",
    "Lagrange Multiplier (LM)"
]

rows = param_rows + stat_rows


# 结果表
sar_table = pd.DataFrame(index=rows, columns=list(weights.keys()))

for w_name, w in weights.items():
    # w_sparse = w.sparse
    # 1. OLS（用于 LM_Error）
    ols_model = OLS(y_arr, X_arr, w=w, spat_diag=True, name_y="ln_access", name_x=list(X.columns))

    # 2. SEM
    # sem_model = ML_Error(y_arr, X_arr, w=w, name_y="ln_access", name_x=list(X.columns))
    # sem_model = ML_Error(y_arr, X_arr, w=w_sparse)
    sar_model = ML_Lag(
        y,
        X,
        w=w,
        name_y="ln_access",
        name_x=list(X.columns)
        # name_x=["const","road1","ln_road2","ln_gdp","ln_price","ln_poi","build"] # for SEM1
        # name_x=["const","road1","ln_road2","ln_price","build"] # for SEM2
    )
    
    

    # ---------- 系数、标准差、显著性 ----------
    # betas 通常包含：const、各解释变量、lambda
    betas = np.asarray(sar_model.betas).flatten()

    # 优先从 vm1 提取标准差（通常包含 lambda）；否则退回 vm/std_err
    # if hasattr(sem_model, "vm1") and sem_model.vm1 is not None: # 注释掉 因为只有2个值
    #     se_all = np.sqrt(np.diag(np.asarray(sem_model.vm1)))
    #     print(se_all)
    # if hasattr(sem_model, "vm") and sem_model.vm is not None:
    #     se_all = np.sqrt(np.diag(np.asarray(sem_model.vm)))
    #     print(se_all)
    if hasattr(sar_model, "std_err"):
        se_all = np.asarray(sar_model.std_err).flatten()
        # print(se_all)
    # else:
        # se_all = np.full(len(betas), np.nan)
    # print("se_all", se_all)
    # z_stat: [(z, p), ...]
    z_stats = sar_model.z_stat if hasattr(sar_model, "z_stat") else [(np.nan, np.nan)] * len(betas)

    for i, row_name in enumerate(param_rows):
        # print("i", i)
        coef = betas[i] if i < len(betas) else np.nan
        se = se_all[i] if i < len(se_all) else np.nan
        p = z_stats[i][1] if i < len(z_stats) else np.nan
        sar_table.loc[row_name, w_name] = f"{coef:.3f}{star(p)}\n({se:.3f})"
    
    if w_name == 'Distance':
        print(1)
        
    
    # ---------- 残差 Moran's I ----------
    resid = np.asarray(sar_model.u).flatten()
    moran_resid = Moran(resid, w)
    sar_table.loc["Residual Moran's I", w_name] = f"{moran_resid.I:.3f}\n[{moran_resid.p_norm:.3f}]"

    # ---------- R2 ----------
    # SEM 通常没有传统 OLS R2，这里优先用 pr2（pseudo R2）
    r2_value = getattr(sar_model, "pr2", np.nan)
    sar_table.loc["R2", w_name] = f"{r2_value:.3f}"

    # ---------- 信息准则 ----------
    logll = getattr(sar_model, "logll", np.nan)
    aic = getattr(sar_model, "aic", np.nan)
    schwarz = getattr(sar_model, "schwarz", np.nan)

    sar_table.loc["Log Likelihood", w_name] = f"{logll:.3f}"
    sar_table.loc["Akaike Info Criterion", w_name] = f"{aic:.3f}"
    # print(f"{aic:.3f}")
    sar_table.loc["Schwarz Criterion", w_name] = f"{schwarz:.3f}"

    # ---------- Wald 统计量 ----------
    # 对 rho 的 Wald 检验：W = (lambda / se_lambda)^2
    rho_idx = len(param_rows) - 1
    rho = betas[rho_idx] if rho_idx < len(betas) else np.nan
    se_rho = se_all[rho_idx] if rho_idx < len(se_all) else np.nan
    wald = (rho / se_rho) ** 2 if pd.notna(rho) and pd.notna(se_rho) and se_rho != 0 else np.nan
    sar_table.loc["Wald", w_name] = f"{wald:.3f}"
    k = X.shape[1]
    wald_p_value = 1 - chi2.cdf(wald, df=k)
    sar_table.loc["Wald_p", w_name] = f"{wald_p_value:.3f}"

    

    # ---------- LR 统计量 ----------
    # 与同一权重矩阵下 OLS 比较：LR = 2 * (LL_sem - LL_ols)
    # OLS 的 logll 手工计算
    n = len(y_arr)
    rss = float(np.sum(np.asarray(ols_model.u).flatten() ** 2))
    ll_ols = -n / 2 * (np.log(2 * np.pi) + 1 + np.log(rss / n))
    lr = 2 * (logll - ll_ols) if pd.notna(logll) else np.nan
    sar_table.loc["Likelihood Ratio (LR)", w_name] = f"{lr:.3f}"

    # ---------- LM 统计量 ----------
    # 采用同一权重矩阵下 OLS 的 LM_Error
    lm = ols_model.lm_error[0] if hasattr(ols_model, "lm_lag") else np.nan
    sar_table.loc["Lagrange Multiplier (LM)", w_name] = f"{lm:.3f}"

# 导出
# sar_table.to_excel("SAR_X2_multiW_5k.xlsx")

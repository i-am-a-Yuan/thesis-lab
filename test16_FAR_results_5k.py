# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:35:40 2026

@author: 86183
"""
# %%


# 改自 OLS_results_5k
import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from spreg import ML_Lag
from esda.moran import Moran
# import spreg
from libpysal.weights import Queen, DistanceBand, KNN
from scipy import stats 


def star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

df2 = pd.read_csv("df2_5k.csv")
# X2 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build" ]] # - ln_pop

# X2 = sm.add_constant(X2)
# X = X2
y = df2["ln_access_5k"]
X_empty = np.zeros((len(y), 1))
X = X_empty

# 读取shp
gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")


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

table = pd.DataFrame()

param_rows = ["W_ln_access"]

stat_rows = [
    "Residual Moran's I",
    "R2",
    "F",
    "Prob(F)",
    "Log Likelihood",
    "Akaike Info Criterion",
    "Schwarz Criterion"
    ]

rows = param_rows + ["R2", "F", "Prob(F)"]
# 结果表
far_table = pd.DataFrame(index=rows, columns=list(weights.keys()))
# %%
for w_name, w in weights.items():
    # model = sm.OLS(y, X).fit()
    # FAR 模型
    far_model = ML_Lag(
        y,
        X,
        w=w,
        slx_lags=0,
        name_y="ln_access",
        name_x=[],
        # FAR 模型特定参数
        regime_err_sep=False,  # 假设误差项没有区域异质性
        regime_lag_sep=False,  # 假设空间滞后项没有区域异质性
        # 可以调整的其他参数
        method='full',  # 使用最大似然估计
        epsilon=1e-05,  # 收敛阈值
        vm=False  # 不计算方差矩阵（如需要可设为True）
    )
    
    ### rho 参数
    # 提取结果
    rho_estimate = far_model.rho
    variance_matrix = far_model.vm
    
    # 计算标准误
    rho_variance = variance_matrix[0, 0]  # 根据实际情况调整索引
    # rho_std_error = np.sqrt(rho_variance)
    rho_std_error = far_model.std_err[0]
    
    # 计算t统计量和p值
    # t_statistic = rho_estimate / rho_std_error
    z_statistic = rho_estimate / rho_std_error
    n = len(y)
    k = 1  # 如果只有rho一个参数
    df = n - k
    # p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
    # p_value = 1 - stats.t.cdf(abs(t_statistic), df) # t 检验
    p_value = 1 - stats.norm.cdf(abs(z_statistic))

    # print(f"Rho估计值: {rho_estimate:.4f}")
    # print(f"标准误: {rho_std_error:.4f}")
    # print(f"t统计量: {t_statistic:.4f}")
    # print(f"p值: {p_value:.4f}")
    
    # betas = np.asarray(far_model.betas).flatten()
    rho = far_model.rho
    p = p_value
    se = rho_std_error
    far_table.loc["W_ln_access"][w_name] = (f"{rho:.3f}{star(p)}\n({se:.3f})")

    # ---------- 残差 Moran's I ----------
    resid = np.asarray(far_model.u).flatten()
    moran_resid = Moran(resid, w)
    far_table.loc["Residual Moran's I", w_name] = f"{moran_resid.I:.3f}\n[{moran_resid.p_norm:.3f}]"

        
    far_table.loc["R2"][w_name] = f"{far_model.pr2:.3f}"
    # far_table.loc["F"][w_name] = f"{model.fvalue:.3f}"
    # far_table.loc["Prob(F)"][w_name] = f"{model.f_pvalue:.3f}"
    
    # ---------- 信息准则 ----------
    logll = getattr(far_model, "logll", np.nan)
    aic = getattr(far_model, "aic", np.nan)
    schwarz = getattr(far_model, "schwarz", np.nan)

    far_table.loc["Log Likelihood", w_name] = f"{logll:.3f}"
    far_table.loc["Akaike Info Criterion", w_name] = f"{aic:.3f}"
    print(f"{aic:.3f}")
    far_table.loc["Schwarz Criterion", w_name] = f"{schwarz:.3f}"
    
    # ---------- Wald 统计量 ----------
    # 对 rho 的 Wald 检验：W = (lambda / se_lambda)^2
    # rho_idx = len(param_rows) - 1
    # rho = betas[rho_idx] if rho_idx < len(betas) else np.nan
    # se_rho = se_all[rho_idx] if rho_idx < len(se_all) else np.nan
    # wald = (rho / se_rho) ** 2 if pd.notna(rho) and pd.notna(se_rho) and se_rho != 0 else np.nan
    # sar_table.loc["Wald", w_name] = f"{wald:.3f}"
    # k = X.shape[1]
    # wald_p_value = 1 - chi2.cdf(wald, df=k)
    # sar_table.loc["Wald_p", w_name] = f"{wald_p_value:.3f}"

    

    # ---------- LR 统计量 ----------
    # 与同一权重矩阵下 OLS 比较：LR = 2 * (LL_sem - LL_ols)
    # OLS 的 logll 手工计算
    # n = len(y)
    # rss = float(np.sum(np.asarray(ols_model.u).flatten() ** 2))
    # ll_ols = -n / 2 * (np.log(2 * np.pi) + 1 + np.log(rss / n))
    # lr = 2 * (logll - ll_ols) if pd.notna(logll) else np.nan
    # sar_table.loc["Likelihood Ratio (LR)", w_name] = f"{lr:.3f}"

    # ---------- LM 统计量 ----------
    # 采用同一权重矩阵下 OLS 的 LM_Error
    # lm = ols_model.lm_error[0] if hasattr(ols_model, "lm_lag") else np.nan
    # sar_table.loc["Lagrange Multiplier (LM)", w_name] = f"{lm:.3f}"
    
    
    # far_table.loc["LM_Lag", name] = f"{model.lm_lag[0]:.3f} ({model.lm_lag[1]:.3f})"
# %%


# far_table.to_excel("FAR_results_multiW_5k.xlsx")

# %%
1

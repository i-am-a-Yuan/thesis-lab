# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:51:31 2026

@author: 86183
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import Queen, KNN
from spreg import ML_Error, OLS

# =========================
# 1. 数据
# =========================
df2 = pd.read_csv("df2.csv")

gdf = gpd.read_file("boundary.shp")
gdf = gdf.merge(df2, on="STREET")

# 如果你前面已经定义了 y，这一行可以删掉
# 推荐这里用 ln_access；若你论文此处用 access，自行改为 df2["access"]
y = df2["ln_access"].values.reshape(-1, 1)

X2 = df2[["road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build"]]

# =========================
# 2. 显著性星号函数
# =========================
def star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

# =========================
# 3. 权重矩阵
# =========================
w_Queen = Queen.from_dataframe(gdf)
w_Queen.transform = "r"

weights = {
    "Queen": w_Queen,
    "K2": KNN.from_dataframe(gdf, k=2),
    "K3": KNN.from_dataframe(gdf, k=3),
    "K4": KNN.from_dataframe(gdf, k=4),
    "K5": KNN.from_dataframe(gdf, k=5),
    "K6": KNN.from_dataframe(gdf, k=6),
    "K7": KNN.from_dataframe(gdf, k=7),
    "K8": KNN.from_dataframe(gdf, k=8),
}

for w in weights.values():
    w.transform = "r"

# =========================
# 4. 行名
#    参数名会尽量按模型返回的 name_x 来取，
#    最后补一个 lambda
# =========================
param_rows = ["CONSTANT", "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build", "lambda"]
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
all_rows = param_rows + stat_rows

result_table = pd.DataFrame(index=all_rows, columns=weights.keys())

# =========================
# 5. 循环估计各 SEM
# =========================
for w_name, w in weights.items():

    # OLS 基准模型：供 LR / LM / R2 使用
    ols = OLS(
        y,
        X2.values,
        w=w,
        spat_diag=True,
        name_y="ln_access",
        name_x=list(X2.columns)
    )

    # SEM（空间误差模型）
    sem = ML_Error(
        y,
        X2.values,
        w=w,
        name_y="ln_access",
        name_x=list(X2.columns)
    )

    # -------------------------
    # 5.1 参数、标准差、p值
    # -------------------------
    # betas 通常包含 constant + 自变量 + lambda
    betas = np.array(sem.betas).flatten()

    # 不同版本 spreg 的标准误属性可能略有不同
    if hasattr(sem, "std_err"):
        std_err = np.array(sem.std_err).flatten()
    elif hasattr(sem, "std_err_vm"):
        std_err = np.array(sem.std_err_vm).flatten()
    else:
        std_err = np.repeat(np.nan, len(betas))

    # z_stat 通常为 [(z, p), ...]
    if hasattr(sem, "z_stat"):
        pvals = [item[1] for item in sem.z_stat]
    else:
        pvals = [np.nan] * len(betas)

    # 行名尽量与模型返回一致
    row_names_model = []
    if hasattr(sem, "name_x") and sem.name_x is not None:
        row_names_model = list(sem.name_x)
    else:
        row_names_model = ["CONSTANT"] + list(X2.columns)

    # 补 lambda
    if len(row_names_model) == len(betas) - 1:
        row_names_model = row_names_model + ["lambda"]
    elif len(row_names_model) < len(betas):
        row_names_model = row_names_model + ["lambda"] * (len(betas) - len(row_names_model))

    # 写入参数结果
    for rn, coef, se, p in zip(row_names_model, betas, std_err, pvals):
        result_table.loc[rn, w_name] = f"{coef:.4f}{star(p)}\n({se:.4f})"

    # -------------------------
    # 5.2 残差 Moran's I
    # -------------------------
    resid = np.array(sem.u).flatten()
    moran_resid = Moran(resid, w)
    result_table.loc["Residual Moran's I", w_name] = f"{moran_resid.I:.4f}[{moran_resid.p_norm:.4f}]"

    # -------------------------
    # 5.3 R2
    # 这里采用 OLS 的 R2，作为对比展示最稳妥
    # 如果你的 spreg 版本有 pr2，可换成 sem.pr2
    # -------------------------
    if hasattr(sem, "pr2"):
        result_table.loc["R2", w_name] = f"{sem.pr2:.4f}"
    else:
        result_table.loc["R2", w_name] = f"{ols.r2:.4f}"

    # -------------------------
    # 5.4 LogLik / AIC / SC
    # -------------------------
    if hasattr(sem, "logll"):
        logll = sem.logll
        result_table.loc["Log Likelihood", w_name] = f"{logll:.4f}"
    else:
        logll = np.nan
        result_table.loc["Log Likelihood", w_name] = ""

    if hasattr(sem, "aic"):
        result_table.loc["Akaike Info Criterion", w_name] = f"{sem.aic:.4f}"
    else:
        result_table.loc["Akaike Info Criterion", w_name] = ""

    if hasattr(sem, "schwarz"):
        result_table.loc["Schwarz Criterion", w_name] = f"{sem.schwarz:.4f}"
    else:
        result_table.loc["Schwarz Criterion", w_name] = ""

    # -------------------------
    # 5.5 Wald（检验 lambda = 0）
    # 用 (lambda / se_lambda)^2
    # -------------------------
    try:
        lambda_coef = betas[-1]
        lambda_se = std_err[-1]
        wald_stat = (lambda_coef / lambda_se) ** 2
        result_table.loc["Wald", w_name] = f"{wald_stat:.4f}"
    except:
        result_table.loc["Wald", w_name] = ""

    # -------------------------
    # 5.6 LR（相对 OLS）
    # LR = 2 * (LL_sem - LL_ols)
    # -------------------------
    if hasattr(ols, "logll") and hasattr(sem, "logll"):
        lr_stat = 2 * (sem.logll - ols.logll)
        result_table.loc["Likelihood Ratio (LR)", w_name] = f"{lr_stat:.4f}"
    else:
        result_table.loc["Likelihood Ratio (LR)", w_name] = ""

    # -------------------------
    # 5.7 LM（采用 OLS 的 LM_Error）
    # -------------------------
    if hasattr(ols, "lm_error"):
        result_table.loc["Lagrange Multiplier (LM)", w_name] = f"{ols.lm_error[0]:.4f}"
    else:
        result_table.loc["Lagrange Multiplier (LM)", w_name] = ""

# =========================
# 6. 导出
# =========================
result_table.to_excel("SEM_X2_multiweights.xlsx")
print(result_table)
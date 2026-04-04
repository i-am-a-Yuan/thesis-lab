# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:12:57 2026

@author: 86183
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm

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

X1 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_pop", "ln_poi", "build" ]]
X2 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build" ]] # - ln_pop
X3 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "build" ]] # - ln_poi
X4 = df2[[ "road1", "ln_road2", "ln_price", "build" ]] # - ln _gdp
X5 = df2[[ "build" ]] # onlyl build
X6 = df2[[ "road1" ]] # only road1

y = df2["ln_access"]

X1 = sm.add_constant(X1)
X2 = sm.add_constant(X2)
X3 = sm.add_constant(X3)
X4 = sm.add_constant(X4)
X5 = sm.add_constant(X5)
X6 = sm.add_constant(X6)



X_dir = {"OLS1":X1, "OLS2": X2, "OLS3": X3, "OLS4": X4, "OLS5":X5, "OLS6":X6}

# table = pd.DataFrame()

rows = list(X1.columns) + ["R2", "F", "Prob(F)"]
table = pd.DataFrame(index=rows, columns=X_dir.keys())

for name, X in X_dir.items():
    model = sm.OLS(y, X).fit()
    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]

        table.loc[var][name] = (f"{coef:.3f}{star(p)}\n({se:.3f})")
    
    table.loc["R2"][name] = f"{model.rsquared:.3f}"
    table.loc["F"][name] = f"{model.fvalue:.3f}"
    table.loc["Prob(F)"][name] = f"{model.f_pvalue:.3f}"
    
    # table.loc["LM_Lag", name] = f"{model.lm_lag[0]:.3f} ({model.lm_lag[1]:.3f})"

table.to_excel("OLS_results_5k.xlsx")


### 单独建立一个文件
for name, X in X_dir.items():
    file_name = name + "_5k_result" + ".xlsx"
    model = sm.OLS(y, X).fit()
    
    rows = []

    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]

        rows.append(f"{coef:.3f}{star(p)}\n({se:.3f})")

    result_df = pd.DataFrame({
        name: rows
    }, index=model.params.index)

    # 加模型统计量
    result_df.loc["R2"] = f"{model.rsquared:.3f}"
    result_df.loc["F"] = f"{model.fvalue:.3f}"
    result_df.loc["Prob(F)"] = f"{model.f_pvalue:.3f}"

    
    # result_df = pd.DataFrame({
    #     "coef": model.params,
    #     "std_err": model.bse,
    #     "t_value": model.tvalues,
    #     "p_value": model.pvalues
    #     })
    result_df.to_excel(file_name)
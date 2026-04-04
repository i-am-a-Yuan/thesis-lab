# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:25:38 2026

@author: 86183
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm



df2 = pd.read_csv("df2.csv")

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



model1 = sm.OLS(y, X1).fit()
result1_df = pd.DataFrame({
    "coef": model1.params,
    "std_err": model1.bse,
    "t_value": model1.tvalues,
    "p_value": model1.pvalues
})

result1_df

X_dir = {"X1":X1, "X2": X2, "X3": X3, "X4": X4}

for name, X in X_dir.items():
    file_name = "OLS_result_" + name + ".xlsx"
    model = sm.OLS(y, X).fit()
    result_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_value": model.tvalues,
        "p_value": model.pvalues
        })
    result_df.to_excel(file_name)

##### 做的好看些

import pandas as pd

def star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

rows = []

for var in model1.params.index:
    coef = model1.params[var]
    se = model1.bse[var]
    p = model1.pvalues[var]

    rows.append(f"{coef:.4f}{star(p)}\n({se:.4f})")

result1_df = pd.DataFrame({
    "Model1": rows
}, index=model1.params.index)

# 加模型统计量
result1_df.loc["R2"] = f"{model1.rsquared:.4f}"
result1_df.loc["F"] = f"{model1.fvalue:.4f}"
result1_df.loc["Prob(F)"] = f"{model1.f_pvalue:.4f}"

print(result1_df)

# 导出Excel
# result1_df.to_excel("regression_table.xlsx")


##### 合并版

X_dir = {"OLS1":X1, "OLS2": X2, "OLS3": X3, "OLS4": X4, "OLS5":X5, "OLS6":X6}

for name, X in X_dir.items():
    file_name = name + "_result" + ".xlsx"
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




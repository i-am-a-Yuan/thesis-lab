# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 20:36:15 2026

@author: 86183
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

df = pd.read_csv("data02.csv")
# df["access"] = df["access_5k"]
df = df.rename(columns={"roads_perc":"road1","roads_nodes":"road2","housing_price":"price","pop_density":"pop","pois":"poi","zone_perc":"build"}); 


# 复制原数据
df2 = df.copy()

# 对数变换
df2["ln_access_5k"] = np.log(df2["access_5k"])
df2["ln_road2"] = np.log(df2["road2"])
df2["ln_gdp"] = np.log(df2["gdp"])
df2["ln_price"] = np.log(df2["price"])
df2["ln_pop"] = np.log(df2["pop"])
df2["ln_poi"] = np.log(df2["poi"])
# df2["ln_build"] = np.log(df2["build"])
df2.to_csv("df2_5k.csv", encoding='utf-8')

######## 处理后

X1 = df2[[ "road1", "ln_road2", "ln_gdp", "ln_price", "ln_pop", "ln_poi", "build"]]
# X1 = df2[[ "ln_road1", "ln_road2", "ln_gdp", "ln_price", "ln_pop", "ln_poi", "ln_build"]]
X1 = sm.add_constant(X1)

y = df2["ln_access_5k"]
# %%


###### VIF 检验1
vif_df = pd.DataFrame()
vif_df["variable"] = X1.columns
vif_df["VIF"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
#vif_df.to_excel("vif1.xlsx")
# 删除常数项
vif_df = vif_df[vif_df["variable"] != "const"]
vif_df
# %%


###### 查看相关系数
# data2 = df2[["access", "road1", "ln_road2", "ln_gdp", "ln_price", "ln_pop", "ln_poi", "build"]]
data2 = df2[["access_5k", "access", "road1", "ln_road2", "ln_gdp", "ln_price", "ln_pop", "ln_poi", "build"]]

corr1 = data2.corr().round(3)
corr1
corr1.to_excel("correlation-all.xlsx")



###### 建立模型1

model1 = sm.OLS(y, X1).fit()
result1_df = pd.DataFrame({
    "coef": model1.params,
    "std_err": model1.bse,
    "t_value": model1.tvalues,
    "p_value": model1.pvalues
})

result1_df
result1_df.to_excel("result2.xlsx")

#####  vif 过大，应该重新做 去除 ln_pop

X2 = df2[["road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build"]]
X2 = sm.add_constant(X2)


###### VIF 检验2
vif2_df = pd.DataFrame()
vif2_df["variable"] = X2.columns
vif2_df["VIF"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif2_df.to_excel("vif2.xlsx")
vif2_df


data2 = df2[["access", "road1", "ln_road2", "ln_gdp", "ln_price", "ln_poi", "build"]]
corr2 = data2.corr().round(3)
corr2

model2 = sm.OLS(y, X2).fit()
result2_df = pd.DataFrame({
    "coef": model2.params,
    "std_err": model2.bse,
    "t_value": model2.tvalues,
    "p_value": model2.pvalues
})

result2_df
result2_df.to_excel("result2.xlsx")


##### vif 过大 去除 ln_build -- 废除

X3 = df2[["road1", "ln_road2", "ln_gdp", "ln_price","ln_pop", "ln_poi"]]
X3 = sm.add_constant(X3)


###### VIF 检验2 -- 废除
vif3_df = pd.DataFrame()
vif3_df["variable"] = X3.columns
vif3_df["VIF"] = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]
vif3_df.to_excel("vif3.xlsx")
vif3_df

data3 = df2[["access", "road1", "ln_road2", "ln_gdp", "ln_price","ln_pop", "ln_poi"]]
corr3 = data3.corr().round(3)
corr3


model3 = sm.OLS(y, X3).fit()
result3_df = pd.DataFrame({
    "coef": model3.params,
    "std_err": model3.bse,
    "t_value": model3.tvalues,
    "p_value": model3.pvalues
})

result3_df


import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
import sys
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
# 切换工作目录
os.chdir(script_dir)

import pandas as pd

df = pd.read_csv("data01.csv")

# 2. 查看列名（确认是否读取正确）
print(df.columns)

# 3. 选取因变量 + 自变量
variables = [
    "access",
    "roads_perc",
    "roads_nodes",
    "gdp",
    "housing_price",
    "pop_density",
    "pois",
    "zone_perc"
]

data = df[variables]

# 4. 生成描述性统计
summary = data.describe()

# 5. 只保留需要的统计量
summary = summary.loc[["mean", "std", "min", "max"]]

print(summary)

# 如果需要保留三位小数
print(summary.round(3))

# 另存为
# 描述性统计
summary01 = data.describe().loc[["mean", "std", "min", "max"]].round(3)

# 导出为 Excel


summary01.to_excel("summary_statistics.xlsx")


# access 前5大值分析
df.sort_values(by="access", ascending=False).head(5)[["STREET", "AREA", "access"]]

df.sort_values(by="access", ascending=False).head(20)[["STREET", "AREA", "access"]]

df.sort_values(by="access", ascending=True).head(5)[["STREET", "AREA", "access"]]

df.sort_values(by="access", ascending=True).head(20)[["STREET", "AREA", "access"]]

# 相关系数
#df[["access","roads_perc","roads_nodes","gdp","housing_price","pop_density","pois","zone_perc"]].corr().round(3).to_excel("correlation.xlsx")

# 更改属性名
df1 = df.rename(columns={"roads_perc":"road1","roads_nodes":"road2","housing_price":"price","pop_density":"pop","pois":"poi","zone_perc":"build"}); 
#下载
#df1[["access","road1","road2","gdp","price","pop","poi","build"]].corr().round(3).to_excel("correlation.xlsx")


# VIF检验

# 自变量（一般不对因变量 access 做VIF，如需也可加入）

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据

# 自变量（不含因变量 access）
X1 = df1[["road1", "road2", "gdp", "price", "pop", "poi", "build"]]

# 加常数项
X2 = sm.add_constant(X1)

# 计算 VIF
vif = pd.DataFrame()
vif["variable"] = X2.columns
vif["VIF"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]

print(vif)
#vif.to_excel("vif.xlsx", index = True)

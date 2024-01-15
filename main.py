
import gurobipy as gp
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
# 公司數量
N = 50
rng = np.random.default_rng(seed=42)
# 期數
MONTH = 60
EXCEL_PATH = "TWN50_Monthly_return_18_22.xlsx"

# ----------------------------------------------------------------------
df = pd.read_excel(Path(__file__).parent/EXCEL_PATH)

# 報酬
company_info = df.loc[:,["name","month","return"]]
company_info.dropna(inplace=True)
company_info = [[name, group] for name, group in df.loc[:,
                                                        ["name", "month", "return"]].groupby("name")]

# 報酬指數
return_index = df.iloc[0:MONTH, [df.columns.get_loc(
    "return_index")-1, df.columns.get_loc("return_index")]]
return_index_dict = return_index.set_index(return_index.iloc[:, 0].map(int))[
    "return_index"].to_dict()

# print(df.iloc[0:N, df.columns.get_loc("BETA")])
# exit()
parameters = df.iloc[0:N, [df.columns.get_loc("name.1"), df.columns.get_loc("碳排放"), df.columns.get_loc("持股比例")]]
parameters.rename(columns={parameters.columns[0]: "name"}, inplace=True)
parameters["name"] = parameters["name"].map(lambda x: x.strip())

for name, group in company_info:
    name_strip = name.strip()
    # use regression to get beta
    x = np.array([return_index_dict[x] for x in group["month"]])
    y = np.array(group["return"])
    beta_value = stats.linregress(x,y).slope
    parameters.loc[parameters.name == name_strip,
                   "BETA"] = beta_value
    # 市場風險
    group["market_risk"] = group["month"].apply(
        lambda x: beta_value*return_index_dict[x])
    # 獨有風險
    group["idiosyncratic_risk"] = group["return"] - group["market_risk"]
    # 獨有風險變異數
    parameters.loc[parameters.name == name_strip,
                   "idiosyncratic_risk_variance"] = group["idiosyncratic_risk"].var(ddof=0)


# 市場因子變異數
market_factor_variance = df.iloc[:MONTH,
                                 df.columns.get_loc("return_index")].var(ddof=0)

# 碳排放量
carbon_emissions = parameters["碳排放"].values
# np.random.default_rng(seed=42)
# carbon_emissions = rng.uniform(size=50)*100

# 成分股比例
Wb = parameters["持股比例"].values
# Wb = np.array([0.02]*50)
Wb = Wb/Wb.sum()
# 碳排放量上限 = 總碳排放量*碳排放目標比例
Qtotal = np.dot(carbon_emissions, Wb)
Q_percentage = df["碳排放目標比例"].values[0]

# Create a new model
m = gp.Model()
# 1*50
Wp = m.addMVar(N)
# 50*50
delta = np.diag(parameters["idiosyncratic_risk_variance"])

# 1*50
B = np.matrix(parameters["BETA"])
# 1*50
Qb = np.matrix(carbon_emissions)

# Set objective function
# 1*50 50*50 50*1
m.setObjective((Wp-Wb)@(B.T@B*market_factor_variance+delta)@(Wp-Wb))
# 碳排放量總和為Qt(數值)
m.addConstr((Qb@Wp.T).sum() == (Qtotal * Q_percentage))
# 投資組合總和等於1
m.addConstr(Wp.sum() == 1)
# Solve it!
m.optimize()

if m.status == gp.GRB.OPTIMAL:
    market_factor_variance = pd.DataFrame(
        [[market_factor_variance]], columns=["market_factor_variance"])
    min_te = pd.DataFrame([np.sqrt(m.ObjVal)], columns=["Min TE"])
    investment_portfolio_sum = pd.DataFrame([np.sum(Wp.X)], columns=["投資組合總和"])
    carbon_emissions_sum = pd.DataFrame([(Qb@Wp.X).sum()], columns=["碳排放量總和"])
    investment_portfolio = pd.DataFrame(Wp.X, columns=["投資組合"])
    carbon_emissions = pd.DataFrame(carbon_emissions, columns=["碳排放量"])
    carbon_emissions_total = pd.DataFrame([Qtotal], columns=["原碳排放量總和"])
    Wb = pd.DataFrame(Wb, columns=["成分股比例"])

    # 輸出主要結果
    pd.concat([min_te, investment_portfolio, investment_portfolio_sum,carbon_emissions_sum,
                       parameters["name"], Wb, carbon_emissions,carbon_emissions_total],axis=1).to_excel(Path(__file__).parent/"result1.xlsx", index=False)
    result = pd.concat([i[1] for i in company_info]).reset_index(drop=True)
    # 輸出其他結果
    result = pd.concat([ result, parameters, return_index, market_factor_variance], axis=1).to_excel(Path(__file__).parent/"result2.xlsx", index=False)
    print("Finished! The file is saved as result1.xlsx, result2.xlsx.")
else:
    print("No solution found")

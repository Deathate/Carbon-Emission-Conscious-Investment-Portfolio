
import gurobipy as gp
import pandas as pd
import numpy as np
from pathlib import Path

# 公司數量
N = 50
rng = np.random.default_rng(seed=42)
# 期數
MONTH = 60
# 成分股比例
Wb = np.array([1/50]*50)
# 碳排放量上限
Q = rng.uniform(size=50)*20
Qtotal = 10
# 碳排放量
carbon_emissions = rng.uniform(size=50)*100
EXCEL_PATH = "TWN50_Monthly_return_18_22.xlsx"

# ----------------------------------------------------------------------
df = pd.read_excel(Path(__file__).parent/EXCEL_PATH)
# 有些地方沒有名稱，所以要先去除
df = df.dropna(subset=["name"])
df["name"] = df["name"].map(lambda x: x.strip())
# 報酬指數
return_index = df.iloc[0:MONTH, [df.columns.get_loc(
    "return_index")-1, df.columns.get_loc("return_index")]]
return_index_dict = return_index.set_index(return_index.iloc[:, 0].map(int))[
    "return_index"].to_dict()
# BETA
beta = df.iloc[0:N, [df.columns.get_loc("BETA")-1, df.columns.get_loc("BETA")]].set_index(
    df.iloc[:, df.columns.get_loc("BETA")-1][0:N].map(lambda x: x.strip()))["BETA"].to_dict()

company_info = [[name, group] for name, group in df.loc[:,
                                                        ["name", "month", "return"]].groupby("name")]
parameters = df.iloc[0:N, [df.columns.get_loc(
    "BETA")-1, df.columns.get_loc("BETA")]]
parameters.rename(columns={parameters.columns[0]: "name"}, inplace=True)
parameters["name"] = parameters["name"].map(lambda x: x.strip())
parameters["idiosyncratic_risk_variance"] = 0.0

for name, group in company_info:
    # 市場風險
    group["market_risk"] = group["month"].apply(
        lambda x, name: beta[name]*return_index_dict[x], args=(name,))
    # 獨有風險
    group["idiosyncratic_risk"] = group["return"] - group["market_risk"]
    # 獨有風險變異數
    parameters.loc[parameters.name == name,
                   "idiosyncratic_risk_variance"] = group["idiosyncratic_risk"].var(ddof=0)

# 市場因子變異數
market_factor_variance = df.iloc[:MONTH,
                                 df.columns.get_loc("return_index")].var(ddof=0)
# 必須為0的投資組合
excluded_investment_portfolio = df["excluded_investment_portfolio"].dropna(
).values.astype(int)
# 投資組合總和下限
wp_lower_bound = df["wp_lower_bound"].values[0]
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
# 碳排放量小於Q
m.addConstr(Qb@Wp.T <= Q)
# 碳排放量總和為Qt(數值)
m.addConstr((Qb@Wp.T).sum() == Qtotal)
# 投資組合總和小於1
m.addConstr(Wp.sum() <= 1)
m.addConstr(Wp.sum() >= wp_lower_bound)
# 某些投資組合必須為0
for i in excluded_investment_portfolio:
    m.addConstr(Wp[i] == 0)
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
    Wb = pd.DataFrame(Wb, columns=["成分股比例"])
    Q = pd.DataFrame(Q, columns=["碳排放量上限"])
    excluded_investment_portfolio = pd.DataFrame(
        excluded_investment_portfolio, columns=["必須為0的投資組合"])
    wp_lower_bound = pd.DataFrame([wp_lower_bound], columns=["投資組合總和下限"])

    # 輸出主要結果
    pd.concat([min_te, investment_portfolio, excluded_investment_portfolio, wp_lower_bound, investment_portfolio_sum,carbon_emissions_sum,
                       parameters["name"], Wb, carbon_emissions, Q],axis=1).to_excel(Path(__file__).parent/"result1.xlsx", index=False)
    result = pd.concat([i[1] for i in company_info]).reset_index(drop=True)
    # 輸出其他結果
    result = pd.concat([ result, parameters, return_index, market_factor_variance], axis=1).to_excel(Path(__file__).parent/"result2.xlsx", index=False)
    print("Finished! The file is saved as result1.xlsx, result2.xlsx.")
else:
    print("No solution found")

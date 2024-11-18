# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xlwings as xl
import scipy.stats as stats
import os
import plotly.graph_objects as go
import datetime
from dateutil.relativedelta import relativedelta

os.chdir(r'P:/Amundi_Milan/Investment/SPECIALISTS/QUANT_RESEARCH/ASSET_ALLOCATION/crossasset/Jung/cbr/')

# %%
data_file = 'cbr.xlsm'

data_macro = pd.read_excel(data_file,sheet_name='input-macro',header=0, index_col=0).dropna()
data_assets = pd.read_excel(data_file,sheet_name='input-assets',header=0, index_col=0).dropna()
cash = data_assets.iloc[:,-1]
cash_mom = cash.pct_change(periods=1).dropna()
# EQ
data_assets = data_assets.iloc[:,0]
# GOV
# data_assets = data_assets.iloc[:,1]
# IG
# data_assets = data_assets.iloc[:,2]
# HY
# data_assets = data_assets.iloc[:,3]
# COMMO
# data_assets = data_assets.iloc[:,4]

delta_macro_yoy = data_macro.pct_change(periods=12).dropna()
delta_assets_yoy = data_assets.pct_change(periods=12).dropna()
delta_assets_mom = data_assets.pct_change(periods=1).dropna()

# %%

r_macro_y, c_macro_y = delta_macro_yoy.shape
df_pctile_macro = pd.DataFrame(index=delta_macro_yoy.index,columns=delta_macro_yoy.columns)
for j in range(c_macro_y):
    for i in range(r_macro_y):
        df_pctile_macro.iloc[i,j] = stats.percentileofscore(delta_macro_yoy.iloc[:i+1,j],delta_macro_yoy.iloc[i,j])/100

# %%
def calc_corr(macro_data: pd.DataFrame, asset_data: pd.DataFrame, asset_idx: int):  
    r1,c1 = macro_data.shape
    # print(asset_data.shape)
    r2, = asset_data.shape
    macro_data=macro_data[-min(r1,r2):]
    # TEMP Solution as we have one more data point for SPX
    asset_data=asset_data[-min(r1,r2):]
    # print(asset_data.shape)
    # print(macro_data.shape)
    corr_pearson = np.zeros((min(r1,r2)-35,c1))
    macro_val = macro_data.values        
    asset_val = asset_data.values
    # print(r2)
 
    for h in range(min(r1,r2)-35):
        # print(f"row{h}")
    
        for i in range(c1):
            # print(asset_val)
            # print(f"var{i}")
            # temp = macro_val[:h,i]
            # print(temp)

            corr_pearson[h,i] = stats.pearsonr(
                macro_val[:h+36,i],
                asset_val[:h+36]).correlation
            # print(corr_pearson[h,i])
    
    return pd.DataFrame(data=corr_pearson, index=asset_data.index[-(min(r1,r2)-35):], columns=macro_data.columns)
corr_spx = calc_corr(macro_data=delta_macro_yoy,asset_data=delta_assets_mom, asset_idx=0)
significant_factors = np.abs(corr_spx) > 0.1
r_corr,c_corr = corr_spx.shape

# %%
low = []; high = []
desc_low = pd.DataFrame(); desc_high = pd.DataFrame()
for i in range(r_corr):
    # point in time split
    df_pctile_macro_subset = df_pctile_macro[df_pctile_macro.index<=corr_spx.index[i]]
    # choose only significant factors (both macro and correlations)
    df_pctile_macro_subset = df_pctile_macro_subset.iloc[:,significant_factors.values[i,:]]
    corr_subset = corr_spx.iloc[i,significant_factors.values[i,:]]
    # multiply 
    corr_spx_pct_df = pd.DataFrame(data=np.multiply(df_pctile_macro_subset.values, np.abs(corr_subset.values.T)), 
                                    index=df_pctile_macro_subset.index,
                                    columns=df_pctile_macro_subset.columns)
    # split ref and comparison pts
    ref_df_pctile_macro_subset = corr_spx_pct_df.iloc[-1,:] 
    hist_df_pctile_macro_subset = corr_spx_pct_df.iloc[:-1,:]
    # calcuate distances 
    distances_full_subset = hist_df_pctile_macro_subset - ref_df_pctile_macro_subset 
    # TODO: np.linalg.norm is in a for loop. Getting the following error when trying to calculate norm for a matrix
    # TODO: TypeError: loop of ufunc does not support argument 0 of type numpy.float64 which has no callable sqrt method
    r,c = distances_full_subset.shape
    distances_norm = []
    for row in range(r):
        distances_norm.append(np.linalg.norm(distances_full_subset.iloc[row]))
    distances_norm = pd.DataFrame(distances_norm, columns = ["L2 Norm"], index = distances_full_subset.index)
    # locate points outside of the bounds
    lower = 10; upper = 100-lower
    thld_low = np.percentile(distances_norm,lower); thld_up = np.percentile(distances_norm,upper)
    low.append(thld_low); high.append(thld_up)
    distance_low = distances_norm[(distances_norm<=thld_low).values]; distance_up = distances_norm[(distances_norm>=thld_up).values]
    # calculate fwd returns
    # shift asset returns forward one month for ease of calculations
    shifted_assets_mom = delta_assets_mom.shift(periods=-1)
    # pick subset 
    shifted_assets_mom = shifted_assets_mom[shifted_assets_mom.index<=corr_spx.index[i]]
    # calculate forward returns
    fwd_returns_low = shifted_assets_mom[shifted_assets_mom.index.isin(distance_low.index)]
    fwd_returns_high = shifted_assets_mom[shifted_assets_mom.index.isin(distance_up.index)]
    # output descriptive of fwd returns
    desc_low = pd.concat([desc_low, fwd_returns_low.describe()], axis=1) 
    desc_high = pd.concat([desc_high, fwd_returns_high.describe()], axis=1)

desc_low.columns = desc_high.columns = corr_spx.index.date

low_df = pd.DataFrame(low, index=corr_spx.index)
high_df = pd.DataFrame(high, index=corr_spx.index)
# %%
# Strategy
def ptf_analytics(total_returns: pd.DataFrame, returns: pd.DataFrame):
    num_yrs = (total_returns.index[-1] - total_returns.index[0])
    num_yrs = num_yrs.days/365
    # print(num_yrs)
    cagr = total_returns.iloc[-1]**(1/num_yrs)-1
    VaR = (1+np.percentile(returns,5))**(12)-1
    CVaR = returns[returns<np.percentile(returns,5)].mean()
    CVaR = (1+CVaR)**(12)-1
    ann_ret = (1+returns.mean())**12-1
    ann_vol = returns.std()*np.sqrt(12)
    cash_ret = (1+cash_mom.mean())**12-1
    sharpe = (ann_ret-cash_ret)/ann_vol
    returns_yearly = pd.DataFrame(returns.values,columns=['Long/Short'])
    returns_yearly["yr"] = corr_spx.index.year
    returns_grouped = (1+returns_yearly.groupby('yr').mean())**12-1
    t = [cagr, ann_ret, ann_vol,sharpe, CVaR, total_returns.iloc[-1]]
    id =  ["CAGR", "Ann. Ret", "Ann. Vol", "Sharpe","CVaR","Total Ret"]
    performance = pd.DataFrame(t, id)
    return performance, returns_grouped


threshold = desc_low.T[["mean"]]
r,c = threshold.shape
# shifted_assets_mom = delta_assets_mom.shift(periods=-1)
avg_fwd_ret = delta_assets_mom.shift(periods=-1).dropna()
avg_fwd_ret = avg_fwd_ret.iloc[-r:].squeeze()
# avg_fwd_ret = pd.DataFrame(avg_fwd_ret)
strategy_lc = []; strategy_ls = []
for i in range(r):
    if (threshold.iloc[i].values > 0):
        strategy_lc.append(avg_fwd_ret.iloc[i])
        strategy_ls.append(avg_fwd_ret.iloc[i])
    else:
        strategy_lc.append(cash_mom.iloc[i])
        strategy_ls.append(-avg_fwd_ret.iloc[i])

strategy_lc = pd.DataFrame(data=strategy_lc, index=desc_low.T.index, columns=[["Long/Cash"]]).dropna()
strategy_ls = pd.DataFrame(data=strategy_ls, index=desc_low.T.index, columns=[["Long/Short"]]).dropna()
strategy_lc_total_ret = (1 + strategy_lc).cumprod()
strategy_ls_total_ret = (1 + strategy_ls).cumprod()
assets_total_ret = (1+ avg_fwd_ret).cumprod()

strategy_lc_perf,strategy_lc_yearly = ptf_analytics(total_returns=strategy_lc_total_ret, returns=strategy_lc)
strategy_ls_perf,strategy_ls_yearly = ptf_analytics(strategy_ls_total_ret, strategy_ls)
assets_perf,asset_yearly = ptf_analytics(assets_total_ret, avg_fwd_ret)
# strategy_cagr = strategy[-1]
output2_strat = pd.concat([strategy_lc_perf, strategy_ls_perf, assets_perf],axis=1)
output2_strat.columns = [["Long/Cash", "Long/Short", delta_assets_mom.name]]
output_strat = pd.concat([strategy_ls.describe(), strategy_ls.describe(), pd.DataFrame(avg_fwd_ret.describe())],axis=1)

output_yearly=pd.concat([strategy_lc_yearly,strategy_ls_yearly,asset_yearly],axis=1)
output_yearly.columns=[["Long/Cash", "Long/Short", delta_assets_mom.name]]


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=strategy_lc_total_ret.index,y=strategy_lc_total_ret.values.flatten(), mode='lines',name="Long/Cash"))
fig.add_trace(go.Scatter(x=strategy_ls_total_ret.index,y=strategy_ls_total_ret.values.flatten(), mode='lines',name="Long/Short"))
fig.add_trace(go.Scatter(x=assets_total_ret.index,y=assets_total_ret.values.flatten(), mode='lines+markers',name="Long Only"))
fig.show()
output_strat
# cbr_strategy(trigger=desc_low, asset_return=delta_assets_mom)
# %%

marker = datetime.datetime.now().strftime("%m-%d-%Y-%H.%M.%S")
with pd.ExcelWriter(data_assets.name+"_"+marker+'.xlsx') as writer:
    corr_spx.to_excel(writer, sheet_name="Correlations")
    desc_low.T.to_excel(writer, sheet_name="Return Stats")
    desc_high.T.to_excel(writer, sheet_name="Return Stats", startcol=12)
    distances_norm.to_excel(writer, sheet_name="Distances (Last)")
    distances_full_subset.to_excel(writer, sheet_name="Distances (Last)", startcol=4)
    delta_assets_mom.to_excel(writer, sheet_name = "data")
    delta_macro_yoy.to_excel(writer, sheet_name = "data", startcol=4)
    output2_strat.to_excel(writer, sheet_name = "PERF")
    strategy_lc_total_ret.to_excel(writer, sheet_name= "PERF", startcol=6)
    strategy_ls_total_ret.to_excel(writer, sheet_name= "PERF", startcol=9)
    assets_total_ret.to_excel(writer, sheet_name="PERF", startcol=12)
    output_yearly.to_excel(writer, sheet_name="PERF", startrow=12)
# def mult_corr_pct(macro_pct: pd.DataFrame, corr: pd.DataFrame):
#     r_corr,c_corr = corr.shape
#     r_macro,c_macro = macro_pct.shape
#     # corr = corr[-min(r_corr,r_macro)]
#     # macro_pct = macro_pct[-min(r_corr,r_macro)]
#     factor_sel = np.abs(corr_spx) > 0.05
#     for i in range(c_corr):

#     return corr_pct_df

# %%

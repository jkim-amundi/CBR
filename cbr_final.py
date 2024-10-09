# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xlwings as xl
import scipy.stats as stats
import os
import plotly.graph_objects as go
os.chdir(r'P:/Amundi_Milan/Investment/SPECIALISTS/QUANT_RESEARCH/ASSET_ALLOCATION/crossasset/Jung/cbr/')

# %%
data_file = 'cbr_hy.xlsm'

data_macro = pd.read_excel(data_file,sheet_name='input-macro',header=0, index_col=0).dropna()
data_assets = pd.read_excel(data_file,sheet_name='input-assets',header=0, index_col=0).dropna()

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
    r2,c2= asset_data.shape
    macro_data=macro_data[-min(r1,r2):]
    # TEMP Solution as we have one more data point for SPX
    asset_data=asset_data[-min(r1,r2):]
    # print(asset_data.shape)
    # print(macro_data.shape)
    corr_pearson = np.zeros((1,c1))
    macro_val = macro_data.values        
    asset_val = asset_data.values
    # print(c1)

    for i in range(c1):
        # print(macro_val[:,i].shape)
        # print(asset_val)
        # print(i)
        corr_pearson[0,i] = stats.pearsonr(macro_val[:,i],asset_val[:,asset_idx]).correlation
    return corr_pearson

corr_spx = calc_corr(macro_data=delta_macro_yoy,asset_data=delta_assets_mom, asset_idx=0)
corr_spx_pct_df = pd.DataFrame(data=np.multiply(df_pctile_macro, np.abs(corr_spx.flatten().T)), 
                               index=df_pctile_macro.index,
                               columns=df_pctile_macro.columns)

# corr_jpmtus = calc_corr(macro_data=macro_y,asset_data=asset_y, asset_idx=1)
# corr_jpmtus_pct_df = pd.DataFrame(data=np.multiply(perc_y, np.abs(corr_jpmtus.flatten().T)), index=macro_y.index,columns=macro_y.columns)


# %%
# asset_m_1mfwd = pd.DataFrame(data=delta_assets_mom.values[1:], index=delta_assets_mom.index[0:-1])

def calculate_fwd_returns(id: int, idx_low: np.array, idx_up: np.array, assets_fwd_ret: int):
    # r,c = idx_low.shape

    r0,c0 = assets_fwd_ret.shape
    r1,c1 = idx_low.shape
    if r0 > r1:
        curr = assets_fwd_ret.iloc[(r0-r1):]
    
    low_spx = curr[idx_low]
    low_spx_pos = low_spx > 0
    low_spx_neg = low_spx < 0
    low_spx_0 = low_spx==0
    up_spx = curr[idx_up]
    low_spx_hit = up_spx.count()

    low_spx = curr[idx_low]
    up_spx = curr[idx_up]
    return low_spx.mean(),up_spx.mean()

def calculate_distance(id: int, threshold: int, corr_pct: pd.DataFrame, asset_idx: int):
    corr_pct_df_sample = corr_pct.iloc[:id-1,:]
    # print(corr_pct.tail())
    to_compare = corr_pct.values[id-1]
    # print(to_compare)
    r_corr,c_corr = corr_pct_df_sample.shape
    hist_dist = np.zeros((r_corr,1))
    distance_factors = np.zeros((r_corr,c_corr))

    for i in range(r_corr):
        hist_dist[i] = np.linalg.norm(corr_pct_df_sample.values[i,:]-to_compare)
        distance_factors[i,:] = corr_pct_df_sample.values[i,:]-to_compare
    distance = pd.DataFrame(data=np.flip(hist_dist), index = corr_pct_df_sample.index[::-1], columns = ['Norm L2'])
    distance_factors_pd = pd.DataFrame(data=distance_factors, index = corr_pct_df_sample.index, columns = data_macro.columns)
    # distance_data = pd.concat([distance,distance_factors_pd],axis=1)
    lower = threshold
    upper = 100-lower
    # print(hist_dist.shape)
    thld_low = np.percentile(hist_dist,lower)
    thld_up = np.percentile(hist_dist,upper)
    idx_low = hist_dist<=thld_low
    idx_up = hist_dist>=thld_up
    distance_low = distance[idx_low] 
    distance_up = distance[idx_up]
    # distance_factors_pd_low = distance_factors_pd[idx_low]
    # distance_factors_pd_up = distance_factors_pd[idx_up]
    # distance_data_low = pd.concat([distance,distance_factors_pd_low], axis=1)
    # distance_data_up = pd.concat([distance,distance_factors_pd_up], axis=1)
    # distance = np.flip(distance)
    # distance = np.flip(distance)
    # distance_low = np.flip(distance_low)
    # distance_up = np.flip(distance_up)
    # idx_low = np.flip(idx_low)
    # idx_up = np.flip(idx_up)
    return distance, distance_low, distance_up, idx_low, idx_up, thld_low, thld_up, distance_factors_pd

# window = 200
# temp2 = corr_spx_pct_df[-window-1:]
# temp3 = temp2.drop(temp2.index[-1])
# returns_spx = pd.DataFrame(index=temp3.index[::-1], columns=[["Below Thld", "Above Thld"]])
# returns_jpmtus = pd.DataFrame(index=temp3.index[::-1], columns=[["Below Thld", "Above Thld"]])
# factors_pd = pd.DataFrame(data=np.zeros((200,3)), index=temp3.index[::-1])
r_1,c_1 = corr_spx_pct_df.shape
min_obs = 59
# returns_spx = pd.DataFrame(index=delta_assets_mom.index[min_obs+11:], columns=[["Below Thld", "Above Thld"]])
returns_spx = np.zeros([r_1-min_obs,2])
j=0
assets2 = pd.DataFrame(data=delta_assets_mom.values[1:],index=delta_assets_mom.index[:-1])
for i in range(-r_1,-min_obs, 1):
    distance_spx, distance_spx_low, distance_spx_up, idx_spx_low, idx_spx_up, thld_low, thld_up, distance_factors_pd = calculate_distance(-i, 10, corr_spx_pct_df,0)
    # print(distance_spx.shape)
    # calculate_fwd_returns(id: int, idx_low: np.array, idx_up: np.array, assets_fwd_ret: int)
    a,b = calculate_fwd_returns(id=i, idx_low=idx_spx_low, idx_up=idx_spx_up, assets_fwd_ret=assets2)
    returns_spx[j,0] = a[0]
    returns_spx[j,1] = b[0]
    j+=1

returns_spx_df = pd.DataFrame(data=np.flip(returns_spx), index=delta_assets_mom.index[-j:], columns=[["Below Thld", "Above Thld"]])
# distance_spx, distance_spx_low, distance_spx_up, idx_spx_low, idx_spx_up = calculate_distance(-i, 10, corr_spx_pct_df,0)
# %%
lower=10
upper=100-lower
distance_spx_curr, distance_spx_low_curr, distance_spx_up_curr, idx_spx_low_curr, idx_spx_up_curr, thld_low, thld_up, distance_factors_pd = calculate_distance(r_1, 10, corr_spx_pct_df,0)
dist_low = distance_spx_curr[np.flip(idx_spx_low_curr,0)]
dist_high = distance_spx_curr[np.flip(idx_spx_up_curr,0)]
title = delta_assets_mom.index[-1].strftime("%d/%m/%Y")
fig = go.Figure()
fig.add_trace(go.Scatter(x=distance_spx_curr.index,y=distance_spx_curr.values.flatten(), mode='lines+markers',name="L2 Distance last"))
fig.add_trace(go.Scatter(x=dist_low.index, 
                         y=dist_low.values.flatten(), mode='markers',name = f'Below {lower}% Threshold'))
fig.add_trace(go.Scatter(x=dist_high.index, y=dist_high.values.flatten(), mode='markers',name = f'Above {upper}% Threshold'))
fig.add_hline(y=thld_low, line_width=1, line_dash="dash", line_color="red")
fig.add_hline(y=thld_up, line_width=1, line_dash="dash", line_color="green")

fig.update_layout(title=f"CBR on {title}")
fig.show()
# %%
wb = xl.books(data_file)
wb.sheets['output'].clear_contents()
wb.sheets['output'].range('a1').value = returns_spx_df
wb.sheets['output'].range('f1').value = returns_spx_df.describe()
wb.sheets['output'].range('n1').value = distance_spx_curr
wb.sheets['output'].range('r1').value = dist_low
wb.sheets['output'].range('u1').value = dist_high
wb.sheets['output2'].range('a1').value = distance_factors_pd

# %%

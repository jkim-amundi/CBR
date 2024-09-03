# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xlwings as xl
import scipy.stats as stats
import os
os.chdir(os.getcwd())

# %%
data_file = 'cbr.xlsx'

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
    
    for i in range(c1):
        # print(macro_val[:,i].shape)
        # print(asset_val[:,i].shape)
        corr_pearson[0,i] = stats.pearsonr(macro_val[:,i],asset_val[:,asset_idx]).correlation
    return corr_pearson

corr_spx = calc_corr(macro_data=delta_macro_yoy,asset_data=delta_assets_yoy, asset_idx=0)
corr_spx_pct_df = pd.DataFrame(data=np.multiply(df_pctile_macro, np.abs(corr_spx.flatten().T)), 
                               index=df_pctile_macro.index,
                               columns=df_pctile_macro.columns)

# corr_jpmtus = calc_corr(macro_data=macro_y,asset_data=asset_y, asset_idx=1)
# corr_jpmtus_pct_df = pd.DataFrame(data=np.multiply(perc_y, np.abs(corr_jpmtus.flatten().T)), index=macro_y.index,columns=macro_y.columns)
# %%
asset_m_1mfwd = pd.DataFrame(data=delta_assets_mom.values[1:], index=delta_assets_mom.index[0:-1])

def calculate_fwd_returns(id: int, idx_low: np.array, idx_up: np.array, asset_idx: int):
    # r,c = idx_low.shape
    curr = asset_m_1mfwd.drop(asset_m_1mfwd.index[id:])
    print(id)
    print(curr.shape)
    low_spx = curr[idx_low]
    up_spx = curr[idx_up]
    return low_spx.values[:,asset_idx].mean(),up_spx.values[:,asset_idx].mean()

def calculate_distance(id: int, threshold: int, corr_pct: pd.DataFrame, asset_idx: int):
    corr_pct_df_sample = corr_pct.iloc[:id,:]

    to_compare = corr_pct.values[-id]

    r_corr,c_corr = corr_pct_df_sample.shape
    hist_dist = np.zeros((r_corr,1))
    distance_factors = np.zeros((r_corr,c_corr))

    for i in range(r_corr):
        hist_dist[i] = np.linalg.norm(corr_pct_df_sample.values[i,:]-to_compare)
        distance_factors[i,:] = corr_pct_df_sample.values[i,:]-to_compare
    distance = pd.DataFrame(data=hist_dist, index = corr_pct_df_sample.index, columns = ['Norm L2'])
    # distance_factors_pd = pd.DataFrame(data=distance_factors, index = corr_pct_df_sample.index, columns = data_def['BBG'])
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
    return distance, distance_low, distance_up, idx_low, idx_up

# window = 200
# temp2 = corr_spx_pct_df[-window-1:]
# temp3 = temp2.drop(temp2.index[-1])
# returns_spx = pd.DataFrame(index=temp3.index[::-1], columns=[["Below Thld", "Above Thld"]])
# returns_jpmtus = pd.DataFrame(index=temp3.index[::-1], columns=[["Below Thld", "Above Thld"]])
# factors_pd = pd.DataFrame(data=np.zeros((200,3)), index=temp3.index[::-1])
r_1,c_1 = corr_spx_pct_df.shape
min_obs = 59
returns_spx = pd.DataFrame(index=delta_assets_mom.index[min_obs:], columns=[["Below Thld", "Above Thld"]])
j=0
for i in range(-r_1,-min_obs, 1):
    distance_spx, distance_spx_low, distance_spx_up, idx_spx_low, idx_spx_up = calculate_distance(-i, 5, corr_spx_pct_df,0)
    returns_spx.iloc[j,0],returns_spx.iloc[j,1] = calculate_fwd_returns(i, idx_spx_low, idx_spx_up, 0)


# %%

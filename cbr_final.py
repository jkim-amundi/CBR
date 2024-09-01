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
        # print(i)
        # print(macro_val[:,i].shape)
        # print(asset_val[:,asset_idx].shape)
        corr_pearson[0,i] = stats.pearsonr(macro_val[:,i],asset_val[:,asset_idx]).correlation
    return corr_pearson

corr_spx = calc_corr(macro_data=df_pctile_macro,asset_data=delta_assets_yoy, asset_idx=0)
corr_spx_pct_df = pd.DataFrame(data=np.multiply(df_pctile_macro, np.abs(corr_spx.flatten().T)), index=delta_assets_yoy.index, columns=df_pctile_macro.columns)


# corr_jpmtus = calc_corr(macro_data=macro_y,asset_data=asset_y, asset_idx=1)
# corr_jpmtus_pct_df = pd.DataFrame(data=np.multiply(perc_y, np.abs(corr_jpmtus.flatten().T)), index=macro_y.index,columns=macro_y.columns)
# %%

# %%
import os
import pandas as pd
import numpy as np
import xlwings as xl
import matplotlib.pyplot as plt
os.chdir(r'P:/Amundi_Milan/Investment/SPECIALISTS/QUANT_RESEARCH/ASSET_ALLOCATION/crossasset/Jung/cbr/')
from pathlib import Path

# %%
import glob
signals = pd.DataFrame()
performance = pd.DataFrame()
path = "P:/Amundi_Milan/Investment/SPECIALISTS/QUANT_RESEARCH/ASSET_ALLOCATION/crossasset/Jung/cbr/*.csv"
for fname in glob.glob(path):
    name = fname.split("\\")[-1].split('_')[0]
    data = pd.read_csv(fname, index_col=0)
    performance[name] = data.iloc[:,-1]
    signals[name] = np.sign(data.iloc[:,0])

# performance['CASH'] = performance.sum(axis=1)
signals['cash'] = -signals.sum(axis=1)
# performance['total'] = performance.sum(axis=1)
# %%
data_file = 'input-assets.csv'
data_assets = pd.read_csv(data_file, header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()
# data_assets = pd.read_excel(data_file,sheet_name='input-assets',header=0, index_col=0).dropna()
data_assets = data_assets.iloc[:,5]
cash_mom = data_assets.pct_change(periods=1).dropna()
cash_mom = cash_mom['2005-04-30':]
damper = 0.05
temp = signals['cash'].shift()
temp['2005-04-30'] = -3
performance['CASH'] = damper * temp.values * cash_mom
# %%
position = pd.DataFrame()
position['RISK'] = signals.COMMO + signals.EQ + signals.HY
position['DURATION'] = signals.IG + signals.GOV
position['CASH'] = signals.cash

# %%
position['RISK ON/N/OFF'] = 'ON'
idx = position.RISK < 0
position.loc[idx,'RISK ON/N/OFF'] = 'OFF'
idx = position.RISK == 0
position.loc[idx,'RISK ON/N/OFF'] = 'NEUTRAL'

position['DURATION L/N/S'] = 'LONG'
idx = position.DURATION < 0
position.loc[idx,'DURATION L/N/S'] = 'SHORT'
idx = position.DURATION == 0
position.loc[idx,'DURATION L/N/S'] = 'NEUTRAL'

position['CASH L/N/S'] = 'ON'
idx = position.CASH < 0
position.loc[idx,'CASH L/N/S'] = 'OFF'
idx = position.CASH == 0
position.loc[idx,'CASH L/N/S'] = 'NEUTRAL'

position['FINAL POSITION'] = position['RISK ON/N/OFF'] + "/" + position['DURATION L/N/S']
position['year'] = pd.to_datetime(position.index).year
# %%
freq = pd.crosstab(position['RISK ON/N/OFF'], position['DURATION L/N/S'], margins=True, dropna=False)
position['FINAL POSITION'].value_counts(normalize=True).plot(kind='barh',stacked=True)

position_grouped = position[['FINAL POSITION','year']].groupby('year').value_counts(normalize=True).unstack()
position_grouped = position_grouped.iloc[::-1]
position_grouped[position_grouped.index > 2018].plot(kind='barh',stacked=True)
plt.legend(loc='best', bbox_to_anchor=(1, 1))
# %%
# importing package
import pandas

# create some data
foo = pandas.Categorical(['a', 'b'], 
                         categories=['a', 'b', 'c'])

bar = pandas.Categorical(['d', 'e'], 
                         categories=['d', 'e', 'f'])

# form crosstab with dropna=True (default)
pandas.crosstab(foo, bar)

# form crosstab with dropna=False
pandas.crosstab(foo, bar, dropna=False)
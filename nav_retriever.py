from ib_insync import *
import pandas as pd, numpy as np
from typing import Optional, Dict, Tuple
from ib_insync import IB, util
import math
import plotly.graph_objects as go
import datetime
import os
import pytz
from util import reduce_all_positions_by_pct

util.startLoop()
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=3)
ib.reqMarketDataType(3)  # delayed ok

# Get Portfolio NAV
av = ib.accountValues()
df_vals = util.df(av)
nav_sgd = df_vals[df_vals.tag == 'NetLiquidation']['value'].astype(float).sum()
account = ib.managedAccounts()[0]

if not np.isfinite(nav_sgd) or abs(nav_sgd) < 1e-6:
    sum_df = util.df(ib.accountSummary(account))
    if (sum_df.tag == 'NetLiquidation').any():
        nav_sgd = float(sum_df.loc[sum_df.tag == 'NetLiquidation', 'value'].iloc[0])
    else:
        raise RuntimeError("Could not retrieve NAV from IBKR.")

# SG time
singapore_tz = pytz.timezone('Asia/Singapore')
raw_dt_utc = datetime.datetime.now(datetime.timezone.utc)
today = pd.Timestamp(raw_dt_utc).tz_convert(singapore_tz).round('min')

new_data = {
    "date": [today],
    "nav": [nav_sgd]
}

stopLossTracker_file = "stopLossTracker.parquet"

new_nav_df = pd.DataFrame(new_data)

if os.path.exists(stopLossTracker_file):
    # Load historical data
    old_df = pd.read_parquet(stopLossTracker_file)
    # Concatenate the existing data with the new observation
    updated_df = pd.concat([old_df, new_nav_df], ignore_index=True)
else:
    # If no file exists, this new DF becomes the starting point
    updated_df = new_nav_df

# Calculate peak to current drawdown
max_nav = updated_df['nav'].max()
current_drawdown = (((max_nav - nav_sgd)/max_nav)*100).round(1)

if current_drawdown > 5:
    print(f"Exceeded 5% from peak: {current_drawdown}")
    reduce_all_positions_by_pct(ib, pct=0.01)
else:
    print(f"Within 5% limit: {current_drawdown}")

updated_df.to_parquet(stopLossTracker_file, index=False)

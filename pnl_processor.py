import pandas as pd

INITIAL_VALUE = 1000000
INPUT_FILE = "../app/nav.parquet"
OUTPUT_FILE = "../app/pnl.parquet"

df = pd.read_parquet(INPUT_FILE)
df.rename(columns={"unrealised" : "unrealisedPnL"}, inplace=True)
df["unrealisedPnL"] = df["unrealisedPnL"].round(2)
df["totalPnL"] = df["nav"] - INITIAL_VALUE
df["realisedPnL"] = (df["nav"] - INITIAL_VALUE) - df["unrealisedPnL"]
df["percentage_change"] = ((df["totalPnL"]/INITIAL_VALUE)*100).round(1)
df.to_parquet(OUTPUT_FILE, index=False)

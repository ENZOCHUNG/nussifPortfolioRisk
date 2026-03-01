import pandas as pd
import os

INITIAL_VALUE = 1000000
INPUT_FILE = "../app/nav.parquet"
OUTPUT_FILE = "../app/pnl.parquet"

# Database module for AWS RDS PostgreSQL
try:
    import db as database
    USE_DATABASE = database.test_connection()
    if USE_DATABASE:
        print("Database connection established - will write to PostgreSQL")
except ImportError:
    USE_DATABASE = False
    print("Database module not found - using parquet files only")

# Load from database if available, otherwise from parquet
if USE_DATABASE:
    try:
        df = database.read_nav()
        print("Loaded NAV data from database")
    except Exception as e:
        print(f"Failed to read from database: {e}, falling back to parquet")
        df = pd.read_parquet(INPUT_FILE)
else:
    df = pd.read_parquet(INPUT_FILE)

df.rename(columns={"unrealised" : "unrealisedPnL"}, inplace=True)
df["unrealisedPnL"] = df["unrealisedPnL"].round(2)
df["totalPnL"] = df["nav"] - INITIAL_VALUE
df["realisedPnL"] = (df["nav"] - INITIAL_VALUE) - df["unrealisedPnL"]
df["percentage_change"] = ((df["totalPnL"]/INITIAL_VALUE)*100).round(1)

# Save to parquet
df.to_parquet(OUTPUT_FILE, index=False)

# Write to database
if USE_DATABASE:
    try:
        database.upsert_pnl(df)
        print("PnL data written to database")
    except Exception as e:
        print(f"Failed to write PnL to database: {e}")

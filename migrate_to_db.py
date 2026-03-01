"""
Migration script to transfer all parquet data to PostgreSQL database.

Run this script once to migrate existing historical data from parquet files
to the AWS RDS PostgreSQL database.

Usage:
    export AWS_RDS_PASSWORD='your_password'
    python migrate_to_db.py
"""

import os
import pandas as pd
from db import (
    init_schema,
    test_connection,
    upsert_nav,
    upsert_pnl,
    upsert_global_avg_metric,
    upsert_weights,
    upsert_vol,
    upsert_corr,
    upsert_stop_loss_tracker,
)


def read_parquet_safe(file_path):
    """Read parquet file with fallback engines."""
    try:
        return pd.read_parquet(file_path, engine='pyarrow')
    except Exception:
        pass
    try:
        return pd.read_parquet(file_path, engine='fastparquet')
    except Exception:
        pass
    return pd.read_parquet(file_path)

PARQUET_FILES = {
    "nav": "nav.parquet",
    "pnl": "pnl.parquet",
    "global_avg_metric": "global_avg_metric.parquet",
    "weights": "weights.parquet",
    "vol": "vol.parquet",
    "corr": "corr.parquet",
    "stop_loss_tracker": "stopLossTracker.parquet",
}


def migrate_nav():
    """Migrate NAV data from parquet to database."""
    file_path = PARQUET_FILES["nav"]
    if not os.path.exists(file_path):
        print(f"  Skipping {file_path} - file not found")
        return 0
    
    df = read_parquet_safe(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Migrating {len(df)} NAV records...")
    upsert_nav(df)
    return len(df)


def migrate_pnl():
    """Migrate PnL data from parquet to database."""
    file_path = PARQUET_FILES["pnl"]
    if not os.path.exists(file_path):
        print(f"  Skipping {file_path} - file not found")
        return 0
    
    df = read_parquet_safe(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Migrating {len(df)} PnL records...")
    upsert_pnl(df)
    return len(df)


def migrate_global_avg_metric():
    """Migrate risk metrics from parquet to database."""
    file_path = PARQUET_FILES["global_avg_metric"]
    if not os.path.exists(file_path):
        print(f"  Skipping {file_path} - file not found")
        return 0
    
    df = read_parquet_safe(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Migrating {len(df)} risk_metrics records...")
    upsert_global_avg_metric(df)
    return len(df)


def migrate_weights():
    """Migrate weights data from parquet to database."""
    file_path = PARQUET_FILES["weights"]
    if not os.path.exists(file_path):
        print(f"  Skipping {file_path} - file not found")
        return 0
    
    df = read_parquet_safe(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Migrating {len(df)} weights records...")
    upsert_weights(df)
    return len(df)


def migrate_vol():
    """Migrate volatility data from parquet to database."""
    file_path = PARQUET_FILES["vol"]
    if not os.path.exists(file_path):
        print(f"  Skipping {file_path} - file not found")
        return 0
    
    df = read_parquet_safe(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Migrating {len(df)} volatility records...")
    upsert_vol(df)
    return len(df)


def migrate_corr():
    """Migrate correlation data from parquet to database."""
    file_path = PARQUET_FILES["corr"]
    if not os.path.exists(file_path):
        print(f"  Skipping {file_path} - file not found")
        return 0
    
    df = read_parquet_safe(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Migrating {len(df)} correlation records...")
    upsert_corr(df)
    return len(df)


def migrate_stop_loss_tracker():
    """Migrate stop loss tracker data from parquet to database."""
    file_path = PARQUET_FILES["stop_loss_tracker"]
    if not os.path.exists(file_path):
        print(f"  Skipping {file_path} - file not found")
        return 0
    
    df = read_parquet_safe(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Migrating {len(df)} stop_loss_tracker records...")
    upsert_stop_loss_tracker(df)
    return len(df)


def run_migration():
    """Run full migration from parquet files to PostgreSQL."""
    print("=" * 60)
    print("PARQUET TO POSTGRESQL MIGRATION")
    print("=" * 60)
    
    print("\n[1/3] Testing database connection...")
    if not test_connection():
        print("ERROR: Cannot connect to database. Check your credentials.")
        print("Make sure AWS_RDS_PASSWORD environment variable is set.")
        return False
    
    print("\n[2/3] Initializing database schema...")
    try:
        init_schema()
    except Exception as e:
        print(f"ERROR: Failed to initialize schema: {e}")
        return False
    
    print("\n[3/3] Migrating data from parquet files...")
    total_records = 0
    
    migrations = [
        ("NAV", migrate_nav),
        ("PnL", migrate_pnl),
        ("Risk Metrics (VaR/ES)", migrate_global_avg_metric),
        ("Weights", migrate_weights),
        ("Volatility", migrate_vol),
        ("Correlation", migrate_corr),
        ("Stop Loss Tracker", migrate_stop_loss_tracker),
    ]
    
    for name, migrate_fn in migrations:
        print(f"\n  [{name}]")
        try:
            count = migrate_fn()
            total_records += count
            print(f"  ✓ {name}: {count} records migrated")
        except Exception as e:
            print(f"  ✗ {name}: Failed - {e}")
    
    print("\n" + "=" * 60)
    print(f"MIGRATION COMPLETE: {total_records} total records migrated")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    run_migration()

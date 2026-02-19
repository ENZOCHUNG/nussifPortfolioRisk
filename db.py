"""
Database connection module for AWS RDS PostgreSQL.

This module provides functions to connect to the PostgreSQL database
and perform CRUD operations for portfolio risk data.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from contextlib import contextmanager
from typing import Optional

DB_CONFIG = {
    "host": os.getenv("AWS_RDS_HOST", "pnl-db.cbukkwwqer7w.ap-southeast-1.rds.amazonaws.com"),
    "port": os.getenv("AWS_RDS_PORT", "5432"),
    "database": os.getenv("AWS_RDS_DATABASE", "postgres"),
    "user": os.getenv("AWS_RDS_USER", "postgres"),
    "password": os.getenv("AWS_RDS_PASSWORD", ""),
}

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Get or create SQLAlchemy engine for PostgreSQL connection."""
    global _engine
    if _engine is None:
        connection_string = (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        _engine = create_engine(connection_string, pool_pre_ping=True)
    return _engine


@contextmanager
def get_connection():
    """Context manager for database connections."""
    engine = get_engine()
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()


def init_schema():
    """Initialize database schema with all required tables."""
    engine = get_engine()
    
    schema_sql = """
    -- NAV time series
    CREATE TABLE IF NOT EXISTS nav (
        id SERIAL PRIMARY KEY,
        date TIMESTAMPTZ NOT NULL UNIQUE,
        nav DECIMAL(15,2) NOT NULL,
        unrealised DECIMAL(15,6)
    );

    -- PnL attribution
    CREATE TABLE IF NOT EXISTS pnl (
        id SERIAL PRIMARY KEY,
        date TIMESTAMPTZ NOT NULL UNIQUE,
        nav DECIMAL(15,2),
        unrealised_pnl DECIMAL(15,2),
        total_pnl DECIMAL(15,2),
        realised_pnl DECIMAL(15,2),
        percentage_change DECIMAL(8,2)
    );

    -- Risk metrics (VaR/ES)
    CREATE TABLE IF NOT EXISTS risk_metrics (
        id SERIAL PRIMARY KEY,
        date TIMESTAMPTZ NOT NULL UNIQUE,
        avg_var99_1d DECIMAL(12,3),
        avg_var99_5d DECIMAL(12,3),
        avg_var99_21d DECIMAL(12,3),
        avg_es99_1d DECIMAL(12,3),
        avg_es99_5d DECIMAL(12,3),
        avg_es99_21d DECIMAL(12,3),
        avg_var95_1d DECIMAL(12,3),
        avg_var95_5d DECIMAL(12,3),
        avg_var95_21d DECIMAL(12,3),
        avg_es95_1d DECIMAL(12,3),
        avg_es95_5d DECIMAL(12,3),
        avg_es95_21d DECIMAL(12,3),
        avg_var90_1d DECIMAL(12,3),
        avg_var90_5d DECIMAL(12,3),
        avg_var90_21d DECIMAL(12,3),
        avg_es90_1d DECIMAL(12,3),
        avg_es90_5d DECIMAL(12,3),
        avg_es90_21d DECIMAL(12,3)
    );

    -- Portfolio weights
    CREATE TABLE IF NOT EXISTS weights (
        id SERIAL PRIMARY KEY,
        date TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        weights DECIMAL(10,6),
        UNIQUE(date, symbol)
    );

    -- Volatility
    CREATE TABLE IF NOT EXISTS volatility (
        id SERIAL PRIMARY KEY,
        date TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        volatility DECIMAL(10,6),
        UNIQUE(date, symbol)
    );

    -- Correlation matrix
    CREATE TABLE IF NOT EXISTS correlation (
        id SERIAL PRIMARY KEY,
        date TIMESTAMPTZ NOT NULL,
        asset1 VARCHAR(20) NOT NULL,
        asset2 VARCHAR(20) NOT NULL,
        corr DECIMAL(15,6),
        UNIQUE(date, asset1, asset2)
    );

    -- Stop-loss tracker
    CREATE TABLE IF NOT EXISTS stop_loss_tracker (
        id SERIAL PRIMARY KEY,
        date TIMESTAMPTZ NOT NULL UNIQUE,
        nav DECIMAL(15,2) NOT NULL
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_nav_date ON nav(date DESC);
    CREATE INDEX IF NOT EXISTS idx_pnl_date ON pnl(date DESC);
    CREATE INDEX IF NOT EXISTS idx_risk_date ON risk_metrics(date DESC);
    CREATE INDEX IF NOT EXISTS idx_weights_date ON weights(date DESC);
    CREATE INDEX IF NOT EXISTS idx_weights_symbol ON weights(symbol);
    CREATE INDEX IF NOT EXISTS idx_vol_date ON volatility(date DESC);
    """
    
    with engine.connect() as conn:
        for statement in schema_sql.split(';'):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        conn.commit()
    
    print("Database schema initialized successfully.")


# ============ READ FUNCTIONS ============

def read_nav() -> pd.DataFrame:
    """Read NAV data from database."""
    query = "SELECT date, nav, unrealised FROM nav ORDER BY date"
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def read_pnl() -> pd.DataFrame:
    """Read PnL data from database."""
    query = """
        SELECT date, nav, unrealised_pnl as "unrealisedPnL", 
               total_pnl as "totalPnL", realised_pnl as "realisedPnL", 
               percentage_change 
        FROM pnl ORDER BY date
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def read_global_avg_metric() -> pd.DataFrame:
    """Read risk metrics (VaR/ES) from database."""
    query = """
        SELECT date,
               avg_var99_1d as "avgVar99_1d", avg_var99_5d as "avgVar99_5d", avg_var99_21d as "avgVar99_21d",
               avg_es99_1d as "avgEs99_1d", avg_es99_5d as "avgEs99_5d", avg_es99_21d as "avgEs99_21d",
               avg_var95_1d as "avgVar95_1d", avg_var95_5d as "avgVar95_5d", avg_var95_21d as "avgVar95_21d",
               avg_es95_1d as "avgEs95_1d", avg_es95_5d as "avgEs95_5d", avg_es95_21d as "avgEs95_21d",
               avg_var90_1d as "avgVar90_1d", avg_var90_5d as "avgVar90_5d", avg_var90_21d as "avgVar90_21d",
               avg_es90_1d as "avgEs90_1d", avg_es90_5d as "avgEs90_5d", avg_es90_21d as "avgEs90_21d"
        FROM risk_metrics ORDER BY date
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def read_weights() -> pd.DataFrame:
    """Read weights data from database."""
    query = "SELECT symbol, weights, date FROM weights ORDER BY date, symbol"
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def read_vol() -> pd.DataFrame:
    """Read volatility data from database."""
    query = "SELECT symbol, volatility, date FROM volatility ORDER BY date, symbol"
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def read_corr() -> pd.DataFrame:
    """Read correlation data from database."""
    query = "SELECT asset1, asset2, corr, date FROM correlation ORDER BY date"
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def read_stop_loss_tracker() -> pd.DataFrame:
    """Read stop loss tracker data from database."""
    query = "SELECT date, nav FROM stop_loss_tracker ORDER BY date"
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


# ============ WRITE FUNCTIONS (UPSERT) ============

def upsert_nav(df: pd.DataFrame):
    """Insert or update NAV data."""
    engine = get_engine()
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO nav (date, nav, unrealised)
                VALUES (:date, :nav, :unrealised)
                ON CONFLICT (date) DO UPDATE SET
                    nav = EXCLUDED.nav,
                    unrealised = EXCLUDED.unrealised
            """), {
                "date": row['date'],
                "nav": row['nav'],
                "unrealised": row.get('unrealised')
            })
        conn.commit()


def upsert_pnl(df: pd.DataFrame):
    """Insert or update PnL data."""
    engine = get_engine()
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO pnl (date, nav, unrealised_pnl, total_pnl, realised_pnl, percentage_change)
                VALUES (:date, :nav, :unrealised_pnl, :total_pnl, :realised_pnl, :percentage_change)
                ON CONFLICT (date) DO UPDATE SET
                    nav = EXCLUDED.nav,
                    unrealised_pnl = EXCLUDED.unrealised_pnl,
                    total_pnl = EXCLUDED.total_pnl,
                    realised_pnl = EXCLUDED.realised_pnl,
                    percentage_change = EXCLUDED.percentage_change
            """), {
                "date": row['date'],
                "nav": row['nav'],
                "unrealised_pnl": row.get('unrealisedPnL'),
                "total_pnl": row.get('totalPnL'),
                "realised_pnl": row.get('realisedPnL'),
                "percentage_change": row.get('percentage_change')
            })
        conn.commit()


def upsert_global_avg_metric(df: pd.DataFrame):
    """Insert or update risk metrics (VaR/ES)."""
    engine = get_engine()
    
    column_map = {
        'avgVar99_1d': 'avg_var99_1d', 'avgVar99_5d': 'avg_var99_5d', 'avgVar99_21d': 'avg_var99_21d',
        'avgEs99_1d': 'avg_es99_1d', 'avgEs99_5d': 'avg_es99_5d', 'avgEs99_21d': 'avg_es99_21d',
        'avgVar95_1d': 'avg_var95_1d', 'avgVar95_5d': 'avg_var95_5d', 'avgVar95_21d': 'avg_var95_21d',
        'avgEs95_1d': 'avg_es95_1d', 'avgEs95_5d': 'avg_es95_5d', 'avgEs95_21d': 'avg_es95_21d',
        'avgVar90_1d': 'avg_var90_1d', 'avgVar90_5d': 'avg_var90_5d', 'avgVar90_21d': 'avg_var90_21d',
        'avgEs90_1d': 'avg_es90_1d', 'avgEs90_5d': 'avg_es90_5d', 'avgEs90_21d': 'avg_es90_21d',
    }
    
    with engine.connect() as conn:
        for _, row in df.iterrows():
            params = {"date": row['date']}
            for old_col, new_col in column_map.items():
                params[new_col] = row.get(old_col)
            
            conn.execute(text("""
                INSERT INTO risk_metrics (
                    date, avg_var99_1d, avg_var99_5d, avg_var99_21d,
                    avg_es99_1d, avg_es99_5d, avg_es99_21d,
                    avg_var95_1d, avg_var95_5d, avg_var95_21d,
                    avg_es95_1d, avg_es95_5d, avg_es95_21d,
                    avg_var90_1d, avg_var90_5d, avg_var90_21d,
                    avg_es90_1d, avg_es90_5d, avg_es90_21d
                )
                VALUES (
                    :date, :avg_var99_1d, :avg_var99_5d, :avg_var99_21d,
                    :avg_es99_1d, :avg_es99_5d, :avg_es99_21d,
                    :avg_var95_1d, :avg_var95_5d, :avg_var95_21d,
                    :avg_es95_1d, :avg_es95_5d, :avg_es95_21d,
                    :avg_var90_1d, :avg_var90_5d, :avg_var90_21d,
                    :avg_es90_1d, :avg_es90_5d, :avg_es90_21d
                )
                ON CONFLICT (date) DO UPDATE SET
                    avg_var99_1d = EXCLUDED.avg_var99_1d,
                    avg_var99_5d = EXCLUDED.avg_var99_5d,
                    avg_var99_21d = EXCLUDED.avg_var99_21d,
                    avg_es99_1d = EXCLUDED.avg_es99_1d,
                    avg_es99_5d = EXCLUDED.avg_es99_5d,
                    avg_es99_21d = EXCLUDED.avg_es99_21d,
                    avg_var95_1d = EXCLUDED.avg_var95_1d,
                    avg_var95_5d = EXCLUDED.avg_var95_5d,
                    avg_var95_21d = EXCLUDED.avg_var95_21d,
                    avg_es95_1d = EXCLUDED.avg_es95_1d,
                    avg_es95_5d = EXCLUDED.avg_es95_5d,
                    avg_es95_21d = EXCLUDED.avg_es95_21d,
                    avg_var90_1d = EXCLUDED.avg_var90_1d,
                    avg_var90_5d = EXCLUDED.avg_var90_5d,
                    avg_var90_21d = EXCLUDED.avg_var90_21d,
                    avg_es90_1d = EXCLUDED.avg_es90_1d,
                    avg_es90_5d = EXCLUDED.avg_es90_5d,
                    avg_es90_21d = EXCLUDED.avg_es90_21d
            """), params)
        conn.commit()


def upsert_weights(df: pd.DataFrame):
    """Insert or update weights data."""
    engine = get_engine()
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO weights (date, symbol, weights)
                VALUES (:date, :symbol, :weights)
                ON CONFLICT (date, symbol) DO UPDATE SET
                    weights = EXCLUDED.weights
            """), {
                "symbol": row['symbol'],
                "weights": row['weights'],
                "date": row['date']
            })
        conn.commit()


def upsert_vol(df: pd.DataFrame):
    """Insert or update volatility data."""
    engine = get_engine()
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO volatility (date, symbol, volatility)
                VALUES (:date, :symbol, :volatility)
                ON CONFLICT (date, symbol) DO UPDATE SET
                    volatility = EXCLUDED.volatility
            """), {
                "symbol": row['symbol'],
                "volatility": row['volatility'],
                "date": row['date']
            })
        conn.commit()


def upsert_corr(df: pd.DataFrame):
    """Insert or update correlation data."""
    engine = get_engine()
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO correlation (date, asset1, asset2, corr)
                VALUES (:date, :asset1, :asset2, :corr)
                ON CONFLICT (date, asset1, asset2) DO UPDATE SET
                    corr = EXCLUDED.corr
            """), {
                "asset1": row['asset1'],
                "asset2": row['asset2'],
                "corr": row['corr'],
                "date": row['date']
            })
        conn.commit()


def upsert_stop_loss_tracker(df: pd.DataFrame):
    """Insert or update stop loss tracker data."""
    engine = get_engine()
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO stop_loss_tracker (date, nav)
                VALUES (:date, :nav)
                ON CONFLICT (date) DO UPDATE SET
                    nav = EXCLUDED.nav
            """), {
                "date": row['date'],
                "nav": row['nav']
            })
        conn.commit()


def test_connection() -> bool:
    """Test database connection."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        print("Database connection successful!")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()

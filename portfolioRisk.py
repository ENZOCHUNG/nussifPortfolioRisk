from ib_insync import *
import pandas as pd, numpy as np
from typing import Optional, Dict, Tuple
from ib_insync import IB, util
import math
import plotly.graph_objects as go
import datetime
import os

# ======== Connection to TWS ========
util.startLoop()
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
ib.reqMarketDataType(3)  # delayed ok

BASE_CCY = 'SGD'
LOOKBACK = '5 Y'
BAR = '1 day'
SUPPORTED_SEC_TYPES = {'STK', 'ETF', 'CASH', 'CFD'}

# ======== Helpers (unchanged from your version unless noted) ========
def _bars_to_series(bars) -> Optional[pd.Series]:
    if not bars:
        return None
    df = util.df(bars)
    s = df.set_index('date')['close'].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s

def _series_direct_fx(pair: str, dur=LOOKBACK, bar=BAR) -> Optional[pd.Series]:
    q = ib.qualifyContracts(Forex(pair))
    if not q:
        return None
    bars = ib.reqHistoricalData(q[0], endDateTime='', durationStr=dur,
                                barSizeSetting=bar, whatToShow='MIDPOINT',
                                useRTH=False, formatDate=1)
    return _bars_to_series(bars)

def get_fx_series(base: str, quote: str, dur=LOOKBACK, bar=BAR):
    """
    Return (Series, src) for BASE/QUOTE using IDEALPRO.
    Avoids CNHUSD/CNHSGD by pivoting through USD with USD on top:
        base/quote = (USD/quote) / (USD/base)
    """
    base = base.upper(); quote = quote.upper()
    if base == quote:
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=250, freq='B')
        return pd.Series(1.0, index=idx), 'IDENTITY'

    # --- CNH special-cases (skip non-existent direct pairs) ---
    if base == 'CNH' and quote == 'USD':
        s = _series_direct_fx('USDCNH', dur, bar)
        return (1.0 / s).rename(None) if s is not None else (None, 'N/A'), 'INVERT:USDCNH'
    if base == 'CNH' and quote == 'SGD':
        s_usd_sgd = _series_direct_fx('USDSGD', dur, bar)
        s_usd_cnh = _series_direct_fx('USDCNH', dur, bar)
        if (s_usd_sgd is not None) and (s_usd_cnh is not None):
            syn = (s_usd_sgd.to_frame('a').join(s_usd_cnh.to_frame('b'), how='inner').eval('a/b').squeeze())
            syn.name = None
            return syn, 'PIVOT:(USDSGD/USDCNH)'

    # Try direct only for pairs likely listed (this avoids CNHSGD/CNHUSD noise)
    if not (base == 'CNH' and quote in ('USD', 'SGD')):
        s = _series_direct_fx(f'{base}{quote}', dur, bar)
        if s is not None:
            return s, f'DIRECT:{base}{quote}'

    # Generic USD pivot (USD on top): base/quote = (USD/quote) / (USD/base)
    s_usd_quote = _series_direct_fx(f'USD{quote}', dur, bar)
    s_usd_base  = _series_direct_fx(f'USD{base}',  dur, bar)
    if (s_usd_quote is not None) and (s_usd_base is not None):
        syn = (s_usd_quote.to_frame('q').join(s_usd_base.to_frame('b'), how='inner').eval('q/b').squeeze())
        syn.name = None
        return syn, f'SYNTH:USD{quote}/USD{base}'

    # Inversions as last resort
    if quote == 'USD':
        inv = _series_direct_fx(f'USD{base}', dur, bar)
        if inv is not None:
            return (1.0 / inv).rename(None), f'INVERT:USD{base}'
    if base == 'USD':
        dir_ = _series_direct_fx(f'USD{quote}', dur, bar)
        if dir_ is not None:
            return dir_.rename(None), f'DIRECT:USD{quote}'
        inv = _series_direct_fx(f'{quote}USD', dur, bar)
        if inv is not None:
            return (1.0 / inv).rename(None), f'INVERT:{quote}USD'
    return None, 'N/A'

def get_equity_series(qc: Contract, dur=LOOKBACK, bar=BAR) -> Optional[pd.Series]:
    bars = ib.reqHistoricalData(qc, endDateTime='', durationStr=dur,
                                barSizeSetting=bar, whatToShow='TRADES',
                                useRTH=False, formatDate=1)
    return _bars_to_series(bars)

def get_equity_last_close_local(qc: Contract) -> Tuple[Optional[float], str]:
    bars = ib.reqHistoricalData(qc, endDateTime='', durationStr='5 D',
                                barSizeSetting='1 day', whatToShow='TRADES',
                                useRTH=False, formatDate=1)
    if not bars:
        return None, "N/A"
    df = util.df(bars)
    return float(df.iloc[-1]['close']), "HMDS:1day TRADES"

def normalize_contract_from_position(p: Position) -> Optional[Contract]:
    c = p.contract
    sec = (getattr(c, 'secType', '') or '').upper()
    sym = getattr(c, 'symbol', None)
    cur = getattr(c, 'currency', None) or 'USD'
    px  = (getattr(c, 'primaryExchange', '') or getattr(c, 'exchange', '') or '').upper()
    if sec == 'CASH':
        return Forex(f'{sym}{cur}') if (sym and cur) else None
    if sec in ('STK', 'ETF'):
        if px == 'SGX' or (getattr(c, 'exchange', '') or '').upper() == 'SGX':
            return Stock(sym, 'SGX', cur)
        if px == 'PINK':
            return Stock(sym, 'PINK', cur)
        if px and px not in ('SMART', 'PINK', 'IDEALPRO'):
            return Stock(sym, 'SMART', cur, primaryExchange=px)
        return Stock(sym, 'SMART', cur)
    if sec == 'CFD':
        return CFD(sym, 'SMART', cur)
    return None

def qualify_safe(contract: Contract) -> Optional[Contract]:
    if contract is None:
        return None
    res = ib.qualifyContracts(contract)
    if res:
        return res[0]
    if isinstance(contract, Stock) and contract.currency == 'USD':
        for venue in ('NASDAQ', 'NYSE', 'ARCA'):
            r2 = ib.qualifyContracts(Stock(contract.symbol, 'SMART', 'USD', primaryExchange=venue))
            if r2:
                return r2[0]
    return None

def _fx_c_per_sgd(ccy: str):
    """
    Price series for ccy/SGD.
    Uses USD pivot to avoid CNHSGD.
    """
    c = ccy.upper()
    if c == BASE_CCY:
        # constant 1 aligned to your existing indices when possible
        if asset_prices_sgd:
            union_index = sorted(set().union(*[s.index for s in asset_prices_sgd.values()]))
            return pd.Series(1.0, index=pd.DatetimeIndex(union_index))
        s_usdsgd, _ = get_fx_series('USD', BASE_CCY)
        return pd.Series(1.0, index=s_usdsgd.index if s_usdsgd is not None else None)

    # Try c/SGD via pivot first (quiet, robust): c/SGD = (USD/SGD) / (USD/c)
    s_usd_sgd = _series_direct_fx('USDSGD', LOOKBACK, BAR)
    s_usd_c   = _series_direct_fx(f'USD{c}', LOOKBACK, BAR)
    if (s_usd_sgd is not None) and (s_usd_c is not None):
        idx = s_usd_sgd.index.intersection(s_usd_c.index)
        if len(idx):
            return (s_usd_sgd.reindex(idx).ffill() / s_usd_c.reindex(idx).ffill()).dropna()

    # Fallback: direct c/SGD (only for currencies where it exists)
    if not (c == 'CNH'):
        s_dir, _ = get_fx_series(c, BASE_CCY)
        if s_dir is not None:
            return s_dir

    return None

# ======== Cash Balances ========
vals = [v for v in ib.accountValues() if v.tag == 'CashBalance']
cash_rows = []
for v in vals:
    if v.currency.upper() == 'BASE':   # skip the synthetic "BASE" line
        continue
    cash_rows.append({
        'symbol': f"{v.currency} CASH",
        'secType': 'CASH',
        'qty': float(v.value),
        'currency': v.currency,
        'last_price_local': 1.0,
        'mv_local': float(v.value),   # cash value itself
        'price_source': 'AccountValues'
    })

# ======== Security Positions ========
SUPPORTED_SEC_TYPES = ('STK','ETF','CFD')  # adjust as needed

positions = [p for p in ib.positions() if (getattr(p.contract, 'secType', '').upper() in SUPPORTED_SEC_TYPES)]

hold_rows = []
for p in positions:
    sec = (getattr(p.contract, 'secType', '') or '').upper()
    sym = p.contract.symbol
    ccy = p.contract.currency
    last_px = None
    src = "N/A"

    if sec in ('STK', 'ETF'):
        base = normalize_contract_from_position(p)
        qc   = qualify_safe(base)
        if qc:
            last_px, src = get_equity_last_close_local(qc)
    elif sec in ('CFD',):
        base, quote = sym, ccy
        s, src_fx = get_fx_series(base, quote)
        if s is not None:
            last_px = float(s.iloc[-1])
            src = src_fx

    hold_rows.append({
        'symbol': sym,
        'secType': sec,
        'qty': float(p.position),
        'currency': ccy,
        'last_price_local': last_px,
        'mv_local': (last_px * p.position) if last_px is not None else None,
        'price_source': src
    })

# ======== Combine Cash + Holdings ========
all_rows = cash_rows + hold_rows
holdings_local_df = pd.DataFrame(all_rows)

print("\n=== Holdings (Local ccy, incl. CASH) ===")
print(holdings_local_df.to_string(index=False))

BASE_CCY = "SGD"

# ======== Build asset price series in SGD (for returns matrix) ========
asset_prices_sgd: Dict[str, pd.Series] = {}
asset_qty: Dict[str, float] = {}
asset_ccy: Dict[str, str] = {}

# ---- Equities / ETFs → price in SGD
for p in positions:
    sec = (getattr(p.contract, 'secType', '') or '').upper()
    if sec not in ('STK', 'ETF'):
        continue
    base = normalize_contract_from_position(p)
    qc   = qualify_safe(base)
    s_loc = get_equity_series(qc)
    if s_loc is None:
        continue
    sym = p.contract.symbol
    ccy = getattr(qc, 'currency', 'USD')
    if ccy == BASE_CCY:
        fx_ccy_sgd = pd.Series(1.0, index=s_loc.index)
    else:
        fx_ccy_sgd, _ = get_fx_series(ccy, BASE_CCY)
        if fx_ccy_sgd is None:
            continue
        fx_ccy_sgd = fx_ccy_sgd.reindex(s_loc.index).ffill()
    price_sgd = (s_loc * fx_ccy_sgd).rename(sym)
    asset_prices_sgd[sym] = price_sgd
    asset_qty[sym] = float(p.position)
    asset_ccy[sym] = ccy

# ======== FX helpers ========
def _align_intersection(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    idx = a.index.intersection(b.index)
    if not len(idx):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return a.reindex(idx).ffill(), b.reindex(idx).ffill()

def _invert(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None or s.empty:
        return None
    return (1.0 / s).replace([np.inf, -np.inf], np.nan).dropna()

def _get_pair(base: str, quote: str) -> Optional[pd.Series]:
    base, quote = base.upper(), quote.upper()
    s, _ = get_fx_series(base, quote)
    if s is not None:
        return s
    s_inv, _ = get_fx_series(quote, base)
    if s_inv is not None:
        return _invert(s_inv)
    return None

def _ratio(a_over_c: pd.Series, b_over_c: pd.Series) -> Optional[pd.Series]:
    a1, b1 = _align_intersection(a_over_c, b_over_c)
    if a1.empty or b1.empty:
        return None
    q = (a1 / b1).dropna()
    return q if not q.empty else None

def _fx_pair(base: str, quote: str) -> Optional[pd.Series]:
    base, quote = base.upper(), quote.upper()
    s = _get_pair(base, quote)
    if s is not None:
        return s
    # USD pivot
    s_base_usd = _get_pair(base, 'USD')
    s_quote_usd = _get_pair(quote, 'USD')
    if s_base_usd is not None and s_quote_usd is not None:
        s_pivot = _ratio(s_base_usd, s_quote_usd)
        if s_pivot is not None:
            return s_pivot
    # SGD pivot
    s_base_sgd = _get_pair(base, BASE_CCY)
    s_quote_sgd = _get_pair(quote, BASE_CCY)
    if s_base_sgd is not None and s_quote_sgd is not None:
        s_pivot2 = _ratio(s_base_sgd, s_quote_sgd)
        if s_pivot2 is not None:
            return s_pivot2
    return None

def _fx_c_per_sgd(ccy: str) -> Optional[pd.Series]:
    c = ccy.upper()
    if c == BASE_CCY:
        if asset_prices_sgd:
            union_index = sorted(set().union(*[s.index for s in asset_prices_sgd.values()]))
            return pd.Series(1.0, index=pd.DatetimeIndex(union_index))
        s_usdsgd = _fx_pair('USD', BASE_CCY)
        idx = s_usdsgd.index if s_usdsgd is not None else None
        return pd.Series(1.0, index=idx)

    s_c_sgd = _fx_pair(c, BASE_CCY)
    if s_c_sgd is None:
        return None

    # sanity check: (c/SGD)*(SGD/c) ≈ 1
    s_sgd_c = _fx_pair(BASE_CCY, c)
    if s_sgd_c is not None:
        a, b = _align_intersection(s_c_sgd, s_sgd_c)
        if len(a) and len(b):
            prod = (a * b).dropna()
            if not len(prod) or not np.isfinite(prod.iloc[-1]) or abs(prod.iloc[-1] - 1.0) > 0.03:
                s_c_sgd = _invert(s_c_sgd)
    return s_c_sgd

def _pair_series(base: str, quote: str) -> Optional[pd.Series]:
    return _fx_pair(base, quote)

# ======== Currency exposures: cash balances + FX pair legs ========

# 1) Pure cash balances
av = ib.accountValues()
cash_balances: Dict[str, float] = {}
for v in av:
    if v.tag != 'CashBalance':
        continue
    ccy = v.currency.upper()
    if ccy == 'BASE':
        ccy = BASE_CCY
    cash_balances[ccy] = cash_balances.get(ccy, 0.0) + float(v.value)

currency_legs_units: Dict[str, float] = {ccy: amt for ccy, amt in cash_balances.items() if ccy != BASE_CCY}

# 2) Add FX pair legs (CASH/CFD)
pair_series_map: Dict[Tuple[str, str], pd.Series] = {}
for p in positions:
    sec = (getattr(p.contract, 'secType', '') or '').upper()
    if sec not in ('CASH', 'CFD'):
        continue
    base = p.contract.symbol.upper()
    quote = p.contract.currency.upper()
    if base == quote:
        continue
    s_bq = _pair_series(base, quote)
    if s_bq is None or s_bq.empty:
        continue
    pair_series_map[(base, quote)] = s_bq

# 3) Leg the pairs (CFD → current-spot currency legs)
for p in positions:
    sec = (getattr(p.contract, 'secType', '') or '').upper()
    if sec != 'CFD':
        continue

    base  = p.contract.symbol.upper()
    quote = p.contract.currency.upper()
    qty   = float(p.position)

    s_bq = _pair_series(base, quote)
    if s_bq is None or s_bq.empty:
        continue
    px_pair = float(s_bq.iloc[-1])

    # base leg: qty units of BASE
    currency_legs_units[base] = currency_legs_units.get(base, 0.0) + qty

    # quote leg: -qty * px_pair units of QUOTE
    currency_legs_units[quote] = currency_legs_units.get(quote, 0.0) - qty * px_pair

    # Generalization:
    # If the CFD's BASE is the same as BASE_CCY (SGD here), then the SGD notional
    # is qty times the *larger* of (BASE/QUOTE) and its inverse.
    #
    # If BASE is not our BASE_CCY, we compute the cash notional in SGD by
    # converting the "contract price" into SGD per contract via current FX.
    #
    if base == BASE_CCY:
        mult = px_pair if abs(px_pair) >= 1.0 else (1.0 / px_pair)
        sgd_notional = qty * mult  # e.g., -75,000 * 5.55 = -416,250 SGD

        # convert SGD → foreign ccy (here: CNH) at current spot
        s_sgd_to_quote = _pair_series(BASE_CCY, quote)  # SGD/quote
        if s_sgd_to_quote is None or s_sgd_to_quote.empty:
            continue
        fx_sgd_to_quote = float(s_sgd_to_quote.iloc[-1])

        # foreign leg is opposite sign to the SGD leg, scaled by current FX
        foreign_units = -sgd_notional * fx_sgd_to_quote  # +2.31m CNH in your example

        # accumulate legs
        currency_legs_units[BASE_CCY] = currency_legs_units.get(BASE_CCY, 0.0) + sgd_notional
        currency_legs_units[quote]    = currency_legs_units.get(quote,    0.0) + foreign_units

    else:
        # BASE is not our portfolio base currency.
        # Compute SGD notional from the pair price and current FX ladders.
        # First, get QUOTE→SGD so we can translate qty * (BASE/QUOTE) into SGD:
        s_quote_to_sgd = _pair_series(quote, BASE_CCY)      # QUOTE/SGD (value of 1 QUOTE in SGD)
        s_base_to_quote = s_bq                               # BASE/QUOTE (contract feed)
        if s_quote_to_sgd is None or s_quote_to_sgd.empty:
            continue

        quote_to_sgd = float(s_quote_to_sgd.iloc[-1])
        base_per_quote = float(s_base_to_quote.iloc[-1])     # BASE/QUOTE

        # contract cash notional in SGD:
        # qty * (BASE/QUOTE) * (QUOTE/SGD) = qty * (BASE/SGD)
        sgd_notional = qty * base_per_quote * quote_to_sgd

        # convert SGD → BASE ccy for the other leg
        s_sgd_to_base = _pair_series(BASE_CCY, base)
        if s_sgd_to_base is None or s_sgd_to_base.empty:
            continue
        fx_sgd_to_base = float(s_sgd_to_base.iloc[-1])

        base_units = -sgd_notional * fx_sgd_to_base

        # accumulate legs
        currency_legs_units[BASE_CCY] = currency_legs_units.get(BASE_CCY, 0.0) + sgd_notional
        currency_legs_units[base]     = currency_legs_units.get(base,     0.0) + base_units


# 4) Create synthetic CCY_<ccy> assets
for ccy, units in currency_legs_units.items():
    if ccy == BASE_CCY:
        continue
    s_c_sgd = _fx_c_per_sgd(ccy)
    if s_c_sgd is None or s_c_sgd.empty:
        continue
    name = f"CCY_{ccy}"
    if asset_prices_sgd:
        union_index = sorted(set().union(*[s.index for s in asset_prices_sgd.values()] + [s_c_sgd.index]))
        s_c_sgd = s_c_sgd.reindex(pd.DatetimeIndex(union_index)).ffill()
    asset_prices_sgd[name] = s_c_sgd.rename(name)
    asset_qty[name] = float(units)
    asset_ccy[name] = BASE_CCY

# ======== NAV retrieval ========
df_vals = util.df(av)
nav_sgd = df_vals[df_vals.tag == 'NetLiquidation']['value'].astype(float).sum()
if not np.isfinite(nav_sgd) or abs(nav_sgd) < 1e-6:
    account = ib.managedAccounts()[0]
    sum_df = util.df(ib.accountSummary(account))
    if (sum_df.tag == 'NetLiquidation').any():
        nav_sgd = float(sum_df.loc[sum_df.tag == 'NetLiquidation', 'value'].iloc[0])
    else:
        raise RuntimeError("Could not retrieve NAV from IBKR.")

# ======== Residual CASH_SGD so ΣMV == NAV ========
_union_index = sorted(set().union(*[s.index for s in asset_prices_sgd.values()])) or pd.date_range(end=pd.Timestamp.today().normalize(), periods=2, freq='D')
asset_prices_sgd['CASH_SGD'] = pd.Series(1.0, index=pd.DatetimeIndex(_union_index), name='CASH_SGD')

_tmp_prices = pd.concat(asset_prices_sgd.values(), axis=1).sort_index().ffill()
last_px_noncash = _tmp_prices.iloc[-1]
noncash_cols = [k for k in asset_prices_sgd.keys() if k != 'CASH_SGD']
noncash_mv = sum(float(last_px_noncash[col]) * float(asset_qty.get(col, 0.0)) for col in noncash_cols)
asset_qty['CASH_SGD'] = float(nav_sgd) - float(noncash_mv)

# ======== Prices → Returns matrix ========
prices = pd.concat(asset_prices_sgd.values(), axis=1).sort_index().dropna(how='all').ffill()
cols = list(prices.columns)
qty_vec = pd.Series({k: asset_qty.get(k, 0.0) for k in cols}).reindex(cols).fillna(0.0)

rets = prices.pct_change().dropna(how='all').fillna(0.0)

# ======== Current MVs & weights vs NAV ========
last_prices = prices.iloc[-1]
mkt_values  = (last_prices * qty_vec).rename('MV_SGD')
weights     = (mkt_values / nav_sgd).fillna(0.0)

# ======== Outputs ========
print("\n=== Market Value in SGD ===")
print(mkt_values.round(2).to_string())
print(f"\nNAV (SGD): {nav_sgd:,.2f}")

print("\n=== Weights (MV/NAV) ===")
print("Weights (sum≈1.0):")
print(weights.round(4).to_string())

print("\nΣMV vs NAV:", float(mkt_values.sum()), float(nav_sgd))
print("Σweights  :", float(weights.sum()))

print("\nPrice matrix shape (T×M):", prices.shape)
print(prices.tail())

print("\nReturns matrix shape (T×M):", rets.shape)
print(rets.tail())

# ======== Portfolio daily returns: r_p = R @ w ========
port_ret_series = rets.dot(weights.reindex(rets.columns).fillna(0.0))
print("\nPortfolio daily return")
print(port_ret_series)

daily_returns = port_ret_series

# cumulative compounded return
cumulative_return = (1 + daily_returns).prod() - 1

# number of days observed
T = daily_returns.shape[0]

# annualized return
annualised_return = (1 + cumulative_return) ** (252 / T) - 1

print("Cumulative return:", cumulative_return)
print("Annualised return:", annualised_return)

daily_vol = daily_returns.std(ddof=1) #1d Vol (5Y sample size)
annualised_vol = daily_vol * np.sqrt(252) 

print("Daily vol:", daily_vol)
print("Annualised vol:", annualised_vol)

class StochasticProcess:
    def __init__(self, drift, vol, delta_t, initial_asset_price):
        self.drift = drift      # annualized drift
        self.vol = vol          # annualized volatility
        self.delta_t = delta_t  # fraction of year (1/252 for daily)
        self.current_asset_price = initial_asset_price
        self.asset_price = [initial_asset_price]

    def time_step(self):
        dw = np.random.normal(0, math.sqrt(self.delta_t))
        ds = (self.drift * self.delta_t + self.vol * dw) * self.current_asset_price
        self.current_asset_price += ds
        self.asset_price.append(self.current_asset_price)

# ===== Simulation parameters =====
dt = 1/252
n_sims = 50000
S0 = nav_sgd

processes_1d = [StochasticProcess(annualised_return, annualised_vol, dt, S0) for _ in range(n_sims)]
for p in processes_1d:
    for _ in range(1):
        p.time_step()

processes_5d = [StochasticProcess(annualised_return, annualised_vol, dt, S0) for _ in range(n_sims)]
for p in processes_5d:
    for _ in range(5):
        p.time_step()

processes_21d = [StochasticProcess(annualised_return, annualised_vol, dt, S0) for _ in range(n_sims)]
for p in processes_21d:
    for _ in range(21):
        p.time_step()

# Collect [S0, S1]
frame_1d = [p.asset_price for p in processes_1d]
frame_5d = [p.asset_price for p in processes_5d]
frame_21d = [p.asset_price for p in processes_21d]

def compute_var_es(pairs, alpha):
    """
    Compute VaR and ES from simulated price paths.
    
    pairs: list of [S0, S1, ... Sk] price paths
    alpha: quantile level (e.g. 0.01 for 99% confidence)
    """
    arr = np.asarray(pairs, dtype=float)
    pnl = arr[:, -1] - arr[:, 0]   # PnL distribution

    # Compute VaR & ES
    var = np.quantile(pnl, alpha)
    es = pnl[pnl <= var].mean()
    return var, es

mcVar99_1d, mcEs99_1d   = compute_var_es(frame_1d, 0.01)   # 99% conf
mcVar95_1d, mcEs95_1d   = compute_var_es(frame_1d, 0.05)   # 95% conf
mcVar90_1d, mcEs90_1d   = compute_var_es(frame_1d, 0.10)   # 90% conf

mcVar99_5d, mcEs99_5d   = compute_var_es(frame_5d, 0.01)
mcVar95_5d, mcEs95_5d   = compute_var_es(frame_5d, 0.05)
mcVar90_5d, mcEs90_5d   = compute_var_es(frame_5d, 0.10)

mcVar99_21d, mcEs99_21d = compute_var_es(frame_21d, 0.01)
mcVar95_21d, mcEs95_21d = compute_var_es(frame_21d, 0.05)
mcVar90_21d, mcEs90_21d = compute_var_es(frame_21d, 0.10)

# ======== declare variables =============
hisVar99_1d = 0
hisVar99_5d = 0
hisVar99_21d = 0
hisEs99_1d = 0
hisEs99_5d = 0
hisEs99_21d = 0

hisVar95_1d = 0
hisVar95_5d = 0
hisVar95_21d = 0
hisEs95_1d = 0
hisEs95_5d = 0
hisEs95_21d = 0

hisVar90_1d = 0
hisVar90_5d = 0
hisVar90_21d = 0
hisEs90_1d = 0
hisEs90_5d = 0
hisEs90_21d = 0

# ======== k-day compounded returns ========
def kday_compounded(returns: pd.Series, k: int) -> pd.Series:
    """Compounded k-day returns from 1D daily returns series"""
    return (1 + returns).rolling(k).apply(np.prod, raw=True) - 1

# ======== VaR / ES from returns ========
def var_es_from_returns(returns: pd.Series, alpha=0, nav_sgd=nav_sgd):
    """Compute Historical VaR% and ES (absolute) from returns series"""
    q = 1 - alpha
    var_pct = np.percentile(returns, 100 * q)  # historical quantile
    es_pct = returns[returns <= var_pct].mean() if any(returns <= var_pct) else var_pct
    return abs(var_pct), abs(es_pct * nav_sgd)

# ======== Horizon VaR/ES ========
def horizon_var_es(port_daily_rets: pd.Series, k: int, alpha=0, nav=nav_sgd):
    if k == 1:
        kret = port_daily_rets
    else:
        kret = kday_compounded(port_daily_rets, k).dropna()
    var_pct, es_abs = var_es_from_returns(kret, alpha=alpha, nav_sgd=nav)
    return var_pct, es_abs

conf_levels = [0.99, 0.95, 0.90]
horizons = [1, 5, 21]

# ======== Save variables into globals with your naming ========
for alpha in conf_levels:
    conf_str = str(int(alpha * 100))   # "99", "95", "90"
    for k in horizons:
        var_pct, es_abs = horizon_var_es(port_ret_series, k, alpha=alpha, nav=nav_sgd)

        var_abs = var_pct * nav_sgd   # absolute VaR in SGD

        # save both VaR and ES
        var_name = f"hisVar{conf_str}_{k}d"   
        es_name  = f"hisEs{conf_str}_{k}d"    

        globals()[var_name] = var_abs
        globals()[es_name]  = es_abs

print("historical 1dVar99:", hisVar99_1d)  
print("historical 5dVar99:", hisVar99_5d)
print("historical 21dVar99:", hisVar99_21d)
print("historical 1dEs99:", hisEs99_1d)  
print("historical 5dEs99:", hisEs99_5d)  
print("historical 21dEs99:", hisEs99_21d)  
print("")
print("historical 1dVar95:", hisVar95_1d)
print("historical 5dVar95:", hisVar95_5d)
print("historical 21dVar95:", hisVar95_21d)
print("historical 1dEs95:", hisEs95_1d)  
print("historical 5dEs95:", hisEs95_5d)  
print("historical 21dEs95:", hisEs95_21d)  
print("")
print("historical 1dVar90:", hisVar90_1d)
print("historical 5dVar90:", hisVar90_5d)
print("historical 21dVar90:", hisVar90_21d)
print("historical 1dEs90:", hisEs90_1d)
print("historical 5dEs90:", hisEs90_5d)
print("historical 21dEs90:", hisEs90_21d)

# Example: today's values
raw_dt = datetime.datetime.now()
today = pd.Timestamp(raw_dt).round('h')

# Put into a 1-row DataFrame
new_row = pd.DataFrame([{
    "date": today,
    "hisVar99_1d" : round(hisVar99_1d, 2),
    "hisVar99_5d" : round(hisVar99_5d, 2),
    "hisVar99_21d" : round(hisVar99_21d, 2),
    "hisEs99_1d" : round(hisEs99_1d, 2),
    "hisEs99_5d" : round(hisEs99_5d, 2),
    "hisEs99_21d" : round(hisEs99_21d, 2), #end 99
    "hisVar95_1d" : round(hisVar95_1d, 2),
    "hisVar95_5d" : round(hisVar95_5d, 2),
    "hisVar95_21d" : round(hisVar95_21d, 2),
    "hisEs95_1d" : round(hisEs95_1d, 2),
    "hisEs95_5d" : round(hisEs95_5d, 2),
    "hisEs95_21d" : round(hisEs95_21d, 2), #end 95
    "hisVar90_1d" : round(hisVar90_1d, 2),
    "hisVar90_5d" : round(hisVar90_5d, 2),
    "hisVar90_21d" : round(hisVar90_21d, 2),
    "hisEs90_1d" : round(hisEs90_1d, 2),
    "hisEs90_5d" : round(hisEs90_5d, 2),
    "hisEs90_21d" : round(hisEs90_21d, 2), #end 90 // end historical
    "mcVar99_1d" : round(-mcVar99_1d, 2),
    "mcVar99_5d" : round(-mcVar99_5d, 2),
    "mcVar99_21d" : round(-mcVar99_21d, 2),
    "mcEs99_1d" : round(-mcEs99_1d, 2),
    "mcEs99_5d" : round(-mcEs99_5d, 2),
    "mcEs99_21d" : round(-mcEs99_21d, 2), # end 99
    "mcVar95_1d" : round(-mcVar95_1d, 2),
    "mcVar95_5d" : round(-mcVar95_5d, 2),
    "mcVar95_21d" : round(-mcVar95_21d, 2),
    "mcEs95_1d" : round(-mcEs95_1d, 2),
    "mcEs95_5d" : round(-mcEs95_5d, 2),
    "mcEs95_21d" : round(-mcEs95_21d, 2), # end 95
    "mcVar90_1d" : round(-mcVar90_1d, 2),
    "mcVar90_5d" : round(-mcVar90_5d, 2),
    "mcVar90_21d" : round(-mcVar90_21d, 2),
    "mcEs90_1d" : round(-mcEs90_1d, 2),
    "mcEs90_5d" : round(-mcEs90_5d, 2),
    "mcEs90_21d" : round(-mcEs90_21d, 2) #end 90 // end monte carlo
}])

print(new_row)

# === Compute averages dynamically ===
avg_dict = {}

# Add date from new_row
avg_dict["date"] = new_row["date"].iloc[0]

for conf in [99, 95, 90]:
    for horizon in [1, 5, 21]:
        # Var
        his_var = new_row[f"hisVar{conf}_{horizon}d"].iloc[0]
        mc_var  = new_row[f"mcVar{conf}_{horizon}d"].iloc[0]
        avg_dict[f"avgVar{conf}_{horizon}d"] = (his_var + mc_var) / 2

        # ES
        his_es = new_row[f"hisEs{conf}_{horizon}d"].iloc[0]
        mc_es  = new_row[f"mcEs{conf}_{horizon}d"].iloc[0]
        avg_dict[f"avgEs{conf}_{horizon}d"] = (his_es + mc_es) / 2

# === Put averages into their own DataFrame ===
avg_df = pd.DataFrame([avg_dict])

print("\n=== Averages DataFrame ===")
print(avg_df)

parquet_file = "global_avg_metricV2.parquet"

# If parquet file exists → load it, else create a new DataFrame
if os.path.exists(parquet_file):
    avg_store = pd.read_parquet(parquet_file)
else:
    avg_store = pd.DataFrame()   # completely empty

# Ensure both sides are datetime
if not avg_store.empty:
    avg_store["date"] = pd.to_datetime(avg_store["date"], errors="coerce")

avg_df["date"] = pd.to_datetime(avg_df["date"])
new_date = avg_df.iloc[0]["date"]

if avg_store.empty:
    # First entry, just assign
    avg_store = avg_df.copy()
elif (avg_store["date"] == new_date).any():
    # Update in place if today's date already exists
    avg_store.loc[avg_store["date"] == new_date, avg_df.columns] = avg_df.values
else:
    # Append if new date
    avg_store = pd.concat([avg_store, avg_df], ignore_index=True)

# Keep file chronological
avg_store = avg_store.sort_values("date").reset_index(drop=True)

# Save back
avg_store.to_parquet(parquet_file, index=False)

# Add a date column to weights (convert Series → DataFrame)
weights_df = weights.to_frame("weights").reset_index().rename(columns={"index": "symbol"})
weights_df["date"] = today

# If file exists, append; else create new
weights_file = "weightsV2.parquet"
if os.path.exists(weights_file):
    old = pd.read_parquet(weights_file)
    weights_df = pd.concat([old, weights_df], ignore_index=True)

weights_df.to_parquet(weights_file, index=False)

# Compute volatility (std dev of returns)
vol_std = rets.std()

# Convert to DataFrame
vol_df = vol_std.to_frame("volatility").reset_index().rename(columns={"index": "symbol"})
vol_df["date"] = today

# File path
vol_file = "volV2.parquet"

# Append if file exists
if os.path.exists(vol_file):
    old = pd.read_parquet(vol_file)
    vol_df = pd.concat([old, vol_df], ignore_index=True)

# Save back
vol_df.to_parquet(vol_file, index=False)

# ======= Correlation Matrix ===========
corr_file = "corrv2.parquet"

def save_correlation(rets, today, corr_file="corrV2.parquet", patterns=("CCY_", "CASH_")):
    """
    Compute correlation matrix (excluding currencies), flatten it,
    and store in a parquet file with date reference.
    If today's date already exists, override it. Else append new.
    """
    # Drop FX (currencies)
    to_drop = [c for c in rets.columns if any(p in c for p in patterns)]
    rets_no_fx = rets.drop(columns=to_drop)

    # Compute correlation
    corr_no_fx = rets_no_fx.corr()

    # Flatten to long format
    corr_long = corr_no_fx.stack().reset_index()
    corr_long.columns = ["asset1", "asset2", "corr"]
    corr_long["date"] = today

    # Save back
    corr_long.to_parquet(corr_file, index=False)
    return corr_long

corr_long = save_correlation(rets, today)
print(corr_long.head())

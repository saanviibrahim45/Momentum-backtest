import pandas as pd
import numpy as np
from pathlib import Path

# Load
monthly_returns = pd.read_parquet("data/processed/monthly_returns.parquet")
long_book       = pd.read_parquet("data/processed/long_book.parquet")
short_book      = pd.read_parquet("data/processed/short_book.parquet")

print(f"Returns: {monthly_returns.shape}, Long book: {long_book.shape}")

# The book formed at end-of-month t earns the return from t to t+1.
# .shift(1) on the book aligns it with next-month returns.
# Fill NaN returns with 0 (delisted/missing stocks contribute nothing).
returns_filled = monthly_returns.fillna(0.0)

long_return  = (long_book.shift(1)  * returns_filled).sum(axis=1)
short_return = (short_book.shift(1) * returns_filled).sum(axis=1)
long_short_return = long_return - short_return

# Drop the warmup period where books are NaN (first 6 months + 1 for shift)
strategy = pd.DataFrame({
    'long':       long_return,
    'short':      short_return,
    'long_short': long_short_return,
}).dropna()

Path("data/processed").mkdir(parents=True, exist_ok=True)
strategy.to_parquet("data/processed/strategy_returns.parquet")

# --- Sanity checks ---
print(f"\nStrategy series: {len(strategy)} months from {strategy.index[0].date()} to {strategy.index[-1].date()}")

print("\n--- Monthly return stats (pre-cost) ---")
print(strategy.describe())

print("\n--- Annualized metrics (pre-cost) ---")
for name in strategy.columns:
    r = strategy[name]
    ann_return = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    print(f"  {name:12s}  Ann.Return={ann_return:+.2%}  Ann.Vol={ann_vol:.2%}  Sharpe={sharpe:.2f}")

print("\n--- Long-short worst 5 months (drawdown check) ---")
print(strategy['long_short'].nsmallest(5))
# Expect: March 2009 momentum crash, April-May 2009 reversal, Nov 2020 vaccine rotation,
# possibly Jan 2021 (GME / short squeeze), Feb-Mar 2020 COVID.
import pandas as pd
import numpy as np
from pathlib import Path

# Load the signal
momentum_signal = pd.read_parquet("data/processed/momentum_signal.parquet")
print(f"Signal loaded: {momentum_signal.shape[0]} months × {momentum_signal.shape[1]} tickers")


def get_decile_weights(signal_row, side='top'):
    """
    For one cross-section (one month-end), return equal weights on the
    top or bottom decile and 0 elsewhere. Output sums to 1.0.
    """
    row = signal_row.dropna()
    if len(row) < 30:                # not enough stocks for deciles to be meaningful
        return pd.Series(0.0, index=signal_row.index)
    decile_size = len(row) // 10     # ~50 stocks per decile when we have ~500 names
    if side == 'top':
        selected = row.nlargest(decile_size).index
    else:
        selected = row.nsmallest(decile_size).index
    w = pd.Series(0.0, index=signal_row.index)
    w.loc[selected] = 1.0 / decile_size
    return w


# Step 1: For each month-end (cohort formation date), get the decile weights.
# Each row sums to 1.0: 1/N on the ~50 names in the decile, 0 elsewhere.
print("Building cohort weights...")
long_cohorts  = momentum_signal.apply(lambda r: get_decile_weights(r, 'top'),    axis=1)
short_cohorts = momentum_signal.apply(lambda r: get_decile_weights(r, 'bottom'), axis=1)

# Step 2: Overlapping portfolios.
# At month-end t, the active long book = average of cohorts formed at t, t-1, ..., t-5.
# Rolling sum of 6 cohorts, divided by 6, gives each cohort 1/6 of total capital.
# min_periods=6 means we wait until 6 cohorts exist before producing a valid book.
print("Combining into overlapping books...")
long_book  = long_cohorts.rolling(6, min_periods=6).sum()  / 6
short_book = short_cohorts.rolling(6, min_periods=6).sum() / 6

# Save
Path("data/processed").mkdir(parents=True, exist_ok=True)
long_book.to_parquet("data/processed/long_book.parquet")
short_book.to_parquet("data/processed/short_book.parquet")

# --- Sanity checks ---
print("\n--- Weights sum to 1.0 after warmup ---")
sums = long_book.sum(axis=1).dropna()
print(f"Long book sum stats: mean={sums.mean():.4f}, min={sums.min():.4f}, max={sums.max():.4f}")

print("\n--- Number of distinct names held over time ---")
n_names = (long_book > 0).sum(axis=1)
print(f"Long book: mean={n_names.mean():.0f} names, range [{n_names.min()}, {n_names.max()}]")
# Expect roughly 150-250: each cohort has ~50 names, six cohorts overlap, with some
# names appearing in multiple consecutive cohorts (high-momentum names persist).

print("\n--- Top 10 long-book holdings at end of 2024 ---")
print(long_book.loc['2024-12-31'].nlargest(10))
# These should be names that were in the top decile across multiple recent months —
# so persistent winners, not just one-month flashes.
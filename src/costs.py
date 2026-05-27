import pandas as pd
import numpy as np
from pathlib import Path

# Load
long_book  = pd.read_parquet("data/processed/long_book.parquet")
short_book = pd.read_parquet("data/processed/short_book.parquet")
strategy   = pd.read_parquet("data/processed/strategy_returns.parquet")


def compute_turnover(book):
    """
    Monthly turnover = half of the L1 distance between consecutive weight vectors.
    Divide by 2 because the L1 distance double-counts (you sell some weight in X,
    you buy the same amount of weight in Y, that's one trade not two).
    """
    diff = (book - book.shift(1)).abs()
    return diff.sum(axis=1) / 2


long_turnover  = compute_turnover(long_book)
short_turnover = compute_turnover(short_book)

print("Average monthly turnover (one-way, per leg):")
print(f"  Long leg:  {long_turnover.mean():.2%}")
print(f"  Short leg: {short_turnover.mean():.2%}")
# Expect ~10-15%/month. Six-cohort overlap implies a theoretical floor near 1/6 ≈ 16.7%
# from the dropping cohort, minus any within-cohort persistence.

# Sensitivity table across cost levels
cost_levels_bps = [0, 10, 20, 30, 50]

print(f"\n{'Cost (round-trip)':>18s} | {'Long Ann.Ret':>13s} | {'Long Sharpe':>12s} | {'L-S Ann.Ret':>12s} | {'L-S Sharpe':>11s}")
print("-" * 80)

for cost_bps in cost_levels_bps:
    cost = cost_bps / 10_000  # convert bps to decimal (20 bps = 0.0020)

    long_net = strategy['long'] - long_turnover.loc[strategy.index] * cost
    # Long-short pays costs on BOTH legs
    ls_net   = strategy['long_short'] - (long_turnover + short_turnover).loc[strategy.index] * cost

    long_ann    = long_net.mean() * 12
    long_sharpe = long_ann / (long_net.std() * np.sqrt(12))
    ls_ann      = ls_net.mean() * 12
    ls_sharpe   = ls_ann / (ls_net.std() * np.sqrt(12))

    label = "Gross" if cost_bps == 0 else f"{cost_bps} bps"
    print(f"{label:>18s} | {long_ann:>12.2%} | {long_sharpe:>12.2f} | {ls_ann:>11.2%} | {ls_sharpe:>11.2f}")

# Save the 20-bps version as the "headline" post-cost strategy
cost = 20 / 10_000
post_cost = pd.DataFrame({
    'long_gross':           strategy['long'],
    'long_net_20bps':       strategy['long'] - long_turnover.loc[strategy.index] * cost,
    'long_short_gross':     strategy['long_short'],
    'long_short_net_20bps': strategy['long_short'] - (long_turnover + short_turnover).loc[strategy.index] * cost,
    'long_turnover':        long_turnover.loc[strategy.index],
    'short_turnover':       short_turnover.loc[strategy.index],
})
post_cost.to_parquet("data/processed/strategy_returns_with_costs.parquet")
print("\nSaved post-cost returns to data/processed/strategy_returns_with_costs.parquet")
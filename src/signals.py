import pandas as pd
from pathlib import Path

# to compute monethy gains subtract the main value from that i think. wiat idk i need to review

# Load the daily prices saved by data.py
prices = pd.read_parquet("data/raw/prices.parquet")
print(f"Loaded daily prices: {prices.shape[0]} days × {prices.shape[1]} tickers")

# Resample to last trading day of each month
# 'BME' = Business Month End (last business day, handles holidays)
monthly_prices = prices.resample('BME').last()
print(f"Monthly prices: {monthly_prices.shape[0]} months × {monthly_prices.shape[1]} tickers")

# Monthly returns — needed downstream for the backtest engine
monthly_returns = monthly_prices.pct_change()

# Momentum signal: 6-month return ending at each month-end
# pct_change(6) computes (price[t] / price[t-6]) - 1
momentum_signal = monthly_prices.pct_change(6)

# Save outputs to data/processed/
Path("data/processed").mkdir(parents=True, exist_ok=True)
monthly_prices.to_parquet("data/processed/monthly_prices.parquet")
monthly_returns.to_parquet("data/processed/monthly_returns.parquet")
momentum_signal.to_parquet("data/processed/momentum_signal.parquet")

# --- Sanity checks ---
print("\n--- Signal distribution (all stocks, all dates) ---")
print(momentum_signal.stack().describe())

print("\n--- Ticker count over time (should rise smoothly from ~400 to ~500) ---")
count_by_date = momentum_signal.notna().sum(axis=1)
print(f"2005-12: {count_by_date.loc['2005-12-31':'2005-12-31'].values}")
print(f"2015-12: {count_by_date.loc['2015-12-31':'2015-12-31'].values}")
print(f"2025-12: {count_by_date.loc['2025-12-31':'2025-12-31'].values}")

print("\n--- Top 5 momentum stocks at end of 2024 (a sanity-check date) ---")
print(momentum_signal.loc['2024-12-31'].nlargest(5))
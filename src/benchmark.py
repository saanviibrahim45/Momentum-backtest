import pandas as pd
import numpy as np

monthly_returns = pd.read_parquet("data/processed/monthly_returns.parquet")
strategy = pd.read_parquet("data/processed/strategy_returns.parquet")

# Equal-weighted S&P 500 benchmark over the same window
ew_return = monthly_returns.mean(axis=1).loc[strategy.index]

ann_return = ew_return.mean() * 12
ann_vol = ew_return.std() * np.sqrt(12)
sharpe = ann_return / ann_vol

print(f"Equal-weighted S&P benchmark: Ann.Return={ann_return:+.2%}  Sharpe={sharpe:.2f}")
print(f"Long leg alpha over EW benchmark: {(strategy['long'].mean() - ew_return.mean()) * 12:+.2%}/year")
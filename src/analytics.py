import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load
strategy        = pd.read_parquet("data/processed/strategy_returns.parquet")
costs           = pd.read_parquet("data/processed/strategy_returns_with_costs.parquet")
monthly_returns = pd.read_parquet("data/processed/monthly_returns.parquet")

# Equal-weighted benchmark over the same window
ew_return = monthly_returns.mean(axis=1).loc[strategy.index]

# Series we want to analyze and chart
series = pd.DataFrame({
    'Long-Only Momentum (gross)':       strategy['long'],
    'Long-Only Momentum (net, 20bps)':  costs['long_net_20bps'],
    'Long-Short Momentum (gross)':      strategy['long_short'],
    'EW S&P Benchmark':                 ew_return,
})


def metrics(returns):
    r = returns.dropna()
    cum = (1 + r).cumprod()
    n_years = len(r) / 12
    cagr     = cum.iloc[-1] ** (1 / n_years) - 1
    ann_vol  = r.std() * np.sqrt(12)
    sharpe   = (r.mean() * 12) / ann_vol if ann_vol > 0 else 0
    downside = r[r < 0].std() * np.sqrt(12)
    sortino  = (r.mean() * 12) / downside if downside > 0 else 0
    dd       = (cum / cum.cummax()) - 1
    max_dd   = dd.min()
    calmar   = cagr / abs(max_dd) if max_dd < 0 else 0
    hit_rate = (r > 0).mean()
    return pd.Series({
        'CAGR': cagr, 'Ann. Vol': ann_vol, 'Sharpe': sharpe,
        'Sortino': sortino, 'Max DD': max_dd, 'Calmar': calmar, 'Hit Rate': hit_rate,
    })

metrics_df = series.apply(metrics).T
print("\n=== Performance Metrics ===")
print(metrics_df.to_string(formatters={
    'CAGR': '{:.2%}'.format, 'Ann. Vol': '{:.2%}'.format, 'Max DD': '{:.2%}'.format,
    'Hit Rate': '{:.2%}'.format, 'Sharpe': '{:.2f}'.format,
    'Sortino': '{:.2f}'.format, 'Calmar': '{:.2f}'.format,
}))

Path("reports").mkdir(exist_ok=True)
metrics_df.to_csv("reports/performance_metrics.csv")

# --- Chart 1: Cumulative returns (log scale) ---
fig, ax = plt.subplots(figsize=(12, 6))
(1 + series).cumprod().plot(ax=ax, linewidth=1.5)
ax.set_title("Cumulative Returns: Momentum Strategy vs. Equal-Weighted Benchmark", fontsize=13)
ax.set_ylabel("Cumulative Return ($1 initial, log scale)")
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/cumulative_returns.png", dpi=150)
plt.close()
print("Saved: reports/cumulative_returns.png")

# --- Chart 2: Rolling 12-month Sharpe ---
fig, ax = plt.subplots(figsize=(12, 5))
for col in ['Long-Only Momentum (net, 20bps)', 'EW S&P Benchmark']:
    rolling = (series[col].rolling(12).mean() * 12) / (series[col].rolling(12).std() * np.sqrt(12))
    rolling.plot(ax=ax, label=col, linewidth=1.5)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(pd.Timestamp('2009-04-30'), color='red',    linestyle='--', alpha=0.5, label='Apr 2009 Momentum Crash')
ax.axvline(pd.Timestamp('2020-11-30'), color='orange', linestyle='--', alpha=0.5, label='Nov 2020 Vaccine Rotation')
ax.set_title("Rolling 12-Month Sharpe Ratio", fontsize=13)
ax.set_ylabel("Sharpe Ratio")
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/rolling_sharpe.png", dpi=150)
plt.close()
print("Saved: reports/rolling_sharpe.png")

# --- Chart 3: Drawdown of the net long-only strategy ---
fig, ax = plt.subplots(figsize=(12, 5))
cum = (1 + series['Long-Only Momentum (net, 20bps)']).cumprod()
dd  = (cum / cum.cummax()) - 1
dd.plot(ax=ax, color='darkred', linewidth=1.2)
ax.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
ax.set_title("Drawdown: Long-Only Momentum (net of 20 bps costs)", fontsize=13)
ax.set_ylabel("Drawdown")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/drawdown.png", dpi=150)
plt.close()
print("Saved: reports/drawdown.png")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

monthly_returns = pd.read_parquet("data/processed/monthly_returns.parquet")
momentum_signal = pd.read_parquet("data/processed/momentum_signal.parquet")


def get_decile_weights(signal_row, side='top'):
    row = signal_row.dropna()
    if len(row) < 30:
        return pd.Series(0.0, index=signal_row.index)
    decile_size = len(row) // 10
    selected = row.nlargest(decile_size).index if side == 'top' else row.nsmallest(decile_size).index
    w = pd.Series(0.0, index=signal_row.index)
    w.loc[selected] = 1.0 / decile_size
    return w


print("Building single-cohort decile weights...")
long_cohorts  = momentum_signal.apply(lambda r: get_decile_weights(r, 'top'),    axis=1)
short_cohorts = momentum_signal.apply(lambda r: get_decile_weights(r, 'bottom'), axis=1)

HORIZON = 60   # months after formation
returns_arr = monthly_returns.fillna(0.0).sort_index()

# Valid formation dates: must have a non-empty cohort AND 60 months of forward data
valid_mask = long_cohorts.sum(axis=1) > 0
formation_dates = long_cohorts.index[valid_mask]
last_valid_idx = len(returns_arr) - HORIZON - 1
formation_dates = [t for t in formation_dates if returns_arr.index.get_loc(t) < last_valid_idx]
print(f"Averaging across {len(formation_dates)} formation dates")

# For each formation date, compute the equal-weighted cohort return at each future event month
winner_paths, loser_paths = [], []

for t in formation_dates:
    idx = returns_arr.index.get_loc(t)
    future = returns_arr.iloc[idx + 1 : idx + 1 + HORIZON]   # months t+1 through t+60

    # Equal-weighted return of the cohort at each future month
    winner_paths.append(future @ long_cohorts.loc[t])
    loser_paths.append(future @ short_cohorts.loc[t])

winner_arr = np.array([p.values for p in winner_paths])    # (n_formation, 60)
loser_arr  = np.array([p.values for p in loser_paths])

# Average monthly return at each event month, across formation dates
mean_winner_ret = winner_arr.mean(axis=0)
mean_loser_ret  = loser_arr.mean(axis=0)
mean_spread_ret = mean_winner_ret - mean_loser_ret

# Cumulative (compounded) from event month 1 to 60
cum_winner = (1 + mean_winner_ret).cumprod() - 1
cum_loser  = (1 + mean_loser_ret).cumprod()  - 1
cum_spread = cum_winner - cum_loser

# --- Chart 1: Replicates the paper's Figure 3 (cumulative spread only) ---
fig, ax = plt.subplots(figsize=(12, 6))
event_months = np.arange(1, HORIZON + 1)
ax.plot(event_months, cum_spread * 100, color='black', linewidth=2)
ax.axvspan(0, 6, alpha=0.15, color='blue',
           label='Holding period (months 1–6)')
ax.axvspan(13, 60, alpha=0.08, color='red',
           label='Post-holding period (months 13–60)')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel("Event Month (months after formation)")
ax.set_ylabel("Cumulative Spread (%)  Winners − Losers")
ax.set_title("Long-Horizon Momentum Returns (Replicates JT 2001 Figure 3)\n"
             "Average cumulative return of winner decile minus loser decile, by months since formation")
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/long_horizon_spread.png", dpi=150)
plt.close()
print("Saved: reports/long_horizon_spread.png")

# --- Chart 2: Both legs separately, for the writeup ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(event_months, cum_winner * 100, color='blue',  linewidth=1.5, label='Winners (top decile)')
ax.plot(event_months, cum_loser  * 100, color='red',   linewidth=1.5, label='Losers (bottom decile)')
ax.plot(event_months, cum_spread * 100, color='black', linewidth=2,   label='Spread (winners − losers)')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel("Event Month")
ax.set_ylabel("Cumulative Return (%)")
ax.set_title("Long-Horizon Returns of Winner and Loser Portfolios")
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/long_horizon_both_legs.png", dpi=150)
plt.close()
print("Saved: reports/long_horizon_both_legs.png")

# Print key event-month values for the writeup
print("\n--- Cumulative spread at key event months ---")
print(f"{'Event Month':>12} | {'Winners':>9} | {'Losers':>9} | {'Spread':>9}")
print("-" * 50)
for k in [3, 6, 12, 24, 36, 48, 60]:
    print(f"{k:>12d} | {cum_winner[k-1]*100:>+8.2f}% | {cum_loser[k-1]*100:>+8.2f}% | {cum_spread[k-1]*100:>+8.2f}%")
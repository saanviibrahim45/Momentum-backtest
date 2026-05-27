import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
import zipfile
import re
from io import BytesIO
from pathlib import Path

# Load strategy returns
strategy = pd.read_parquet("data/processed/strategy_returns.parquet")
costs    = pd.read_parquet("data/processed/strategy_returns_with_costs.parquet")


def fetch_ken_french(url):
    """Download a Ken French CSV zip and parse the monthly factor section."""
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(BytesIO(r.content))
    text = z.read(z.namelist()[0]).decode('utf-8')

    factor_names = {'Mkt-RF', 'SMB', 'HML', 'RF', 'Mom', 'WML'}
    header = None
    monthly_rows = []

    for line in text.split('\n'):
        parts = [p.strip() for p in line.split(',')]
        # Data row: first cell is YYYYMM (6 digits)
        if parts and re.match(r'^\d{6}$', parts[0]):
            monthly_rows.append(parts)
        # Header row: at least one cell exactly matches a known factor name
        elif header is None and any(p in factor_names for p in parts):
            header = parts
            header[0] = 'Date'   # first column is unlabeled in the CSV

    df = pd.DataFrame(monthly_rows, columns=header)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m').dt.to_period('M')
    df.set_index('Date', inplace=True)
    return df.apply(pd.to_numeric) / 100

    df = pd.DataFrame(monthly_rows, columns=header)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m').dt.to_period('M')
    df.set_index('Date', inplace=True)
    return df.apply(pd.to_numeric) / 100   # CSV is in percent, convert to decimal


print("Fetching Fama-French factors from Ken French's data library...")
ff3 = fetch_ken_french("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip")
mom = fetch_ken_french("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip")
mom.columns = ['UMD']

factors = ff3.join(mom).loc['2003':'2026']
print(f"Factor columns: {factors.columns.tolist()}")
print(f"Sample: {factors.index[0]} to {factors.index[-1]}, {len(factors)} months\n")


def run_regression(returns, factors, factor_cols):
    df = pd.concat([returns.rename('ret'), factors], axis=1).dropna()
    excess = df['ret'] - df['RF']
    X = sm.add_constant(df[factor_cols])
    return sm.OLS(excess, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})


series = {
    'Long-Only (gross)':      strategy['long'],
    'Long-Only (net, 20bps)': costs['long_net_20bps'],
    'Long-Short (gross)':     strategy['long_short'],
}
for k, v in series.items():
    v.index = v.index.to_period('M')

print(f"{'Strategy':<25s} | {'Model':<3s} | {'Alpha (ann.)':>13s} | {'t-stat':>7s} | {'R²':>6s}")
print("-" * 75)
for name, ret in series.items():
    for model_name, cols in [('3F', ['Mkt-RF', 'SMB', 'HML']),
                             ('4F', ['Mkt-RF', 'SMB', 'HML', 'UMD'])]:
        m = run_regression(ret, factors, cols)
        alpha_ann = m.params['const'] * 12
        print(f"{name:<25s} | {model_name:<3s} | {alpha_ann:>+12.2%} | {m.tvalues['const']:>+7.2f} | {m.rsquared:>6.3f}")

print("\n=== Detailed factor loadings: Long-Only (net, 20bps) ===")
for model_name, cols in [('3-Factor (Fama-French)', ['Mkt-RF', 'SMB', 'HML']),
                          ('4-Factor (FF + Momentum UMD)', ['Mkt-RF', 'SMB', 'HML', 'UMD'])]:
    m = run_regression(series['Long-Only (net, 20bps)'], factors, cols)
    print(f"\n{model_name}:")
    for var in m.params.index:
        coef, t = m.params[var], m.tvalues[var]
        if var == 'const':
            print(f"  Alpha        : {coef*12:+.2%}/yr   (t={t:+.2f})")
        else:
            print(f"  {var:<12s} : {coef:+.4f}        (t={t:+.2f})")
    print(f"  R-squared    : {m.rsquared:.3f}")

Path("reports").mkdir(exist_ok=True)
with open("reports/attribution_results.txt", "w") as f:
    for name, ret in series.items():
        f.write(f"\n{'='*70}\n{name}\n{'='*70}\n")
        for model_name, cols in [('3-Factor (FF)', ['Mkt-RF', 'SMB', 'HML']),
                                  ('4-Factor (FF + UMD)', ['Mkt-RF', 'SMB', 'HML', 'UMD'])]:
            m = run_regression(ret, factors, cols)
            f.write(f"\n{model_name}:\n{m.summary()}\n")
print("\nFull regression results saved to reports/attribution_results.txt")
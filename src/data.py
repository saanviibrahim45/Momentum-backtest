import yfinance as yf
import pandas as pd
from pathlib import Path

# load tickers from the CSV you downloaded
sp500 = pd.read_csv("data/raw/sp500_companies.csv")  # adjust path if you put it elsewhere
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
print(f"Got {len(tickers)} tickers. First few: {tickers[:5]}")

# download price data
data = yf.download(
    tickers,
    start="2003-01-01",
    end="2026-05-24",
    auto_adjust=True,
    progress=True,
)
prices = data['Close']

# saveing
Path("data/raw").mkdir(parents=True, exist_ok=True)
prices.to_parquet("data/raw/prices.parquet")
print(f"Saved prices: {prices.shape[0]} days × {prices.shape[1]} tickers")
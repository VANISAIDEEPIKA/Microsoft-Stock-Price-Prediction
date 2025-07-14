# ---------------- helpers/data_loader.py ----------------
import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start, end):
    """Load historical stock data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {ticker} between {start} and {end}.")

    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# ---------------- helpers/indicators.py ----------------
import pandas as pd

def compute_rsi(df, window=14):
    """Calculate Relative Strength Index (RSI)."""
    data = df.copy()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def compute_macd(df, short=12, long=26, signal=9):
    """Calculate MACD and Signal Line."""
    data = df.copy()
    data['EMA12'] = data['Close'].ewm(span=short, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=long, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    return data

import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px

def plot_predictions(train, valid):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train['Date'], train['Close'], label='Train')
    ax.plot(valid['Date'], valid['Close'], label='Actual')
    ax.plot(valid['Date'], valid['Predictions'], label='Predicted')
    ax.set_title('MSFT Stock Price Prediction')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    return fig

def plot_rsi_macd(df):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    ax[0].plot(df['Date'], df['RSI'], label='RSI', color='blue')
    ax[0].axhline(70, color='red', linestyle='--')
    ax[0].axhline(30, color='green', linestyle='--')
    ax[0].set_title('RSI')
    ax[0].legend()
    ax[0].grid(True)
    
    ax[1].plot(df['Date'], df['MACD'], label='MACD', color='purple')
    ax[1].plot(df['Date'], df['Signal'], label='Signal', color='orange')
    ax[1].set_title('MACD')
    ax[1].legend()
    ax[1].grid(True)

    return fig

def build_future_df(last_date, preds):
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(preds), freq='B')
    return pd.DataFrame({'Date': future_dates, 'Predicted': preds})

def plot_forecast_range_bar(low_price, avg_price, high_price):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[low_price, avg_price - low_price, high_price - avg_price],
        y=['Target Range'],
        orientation='h',
        marker=dict(color=['red', 'gray', 'green']),
        text=[f"Low: ${low_price}", f"Avg: ${avg_price}", f"High: ${high_price}"],
        hoverinfo='text'
    ))
    fig.update_layout(title="📊 Forecast Price Range (Low → High)", height=150)
    return fig

def plot_unified_forecast_line(df_actual, df_future):
    combined_df = pd.concat([
        df_actual[['Date', 'Close']].iloc[-100:],  # Last 100 days
        pd.DataFrame({'Date': df_future['Date'], 'Close': df_future['Predicted']})
    ], ignore_index=True)
    
    fig = px.line(combined_df, x='Date', y='Close', title="📈 Historical + Forecasted MSFT Prices")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
    return fig

def plot_analyst_pie_chart(ratings_dict):
    fig = px.pie(values=ratings_dict.values(), names=ratings_dict.keys(),
                 title="🧠 Analyst Ratings (Mock Data)")
    return fig

# ---------------- app.py ----------------
import streamlit as st
import pandas as pd
from datetime import date
import numpy as np

from helpers.data_loader import load_stock_data
from helpers.indicators import compute_rsi, compute_macd
from helpers.model import scale_data, create_sequences, build_model, predict_future
from helpers.utils import (
    plot_predictions,
    plot_rsi_macd,
    build_future_df,
    plot_forecast_range_bar,
    plot_unified_forecast_line,
    plot_analyst_pie_chart
)

st.set_page_config(page_title="📈 MSFT Stock Predictor", layout="wide")
st.title("🔮 Microsoft Stock Price Predictor")
st.markdown("Built with Streamlit, BiLSTM, and Technical Indicators")

# 📅 Sidebar Inputs
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(date.today().replace(year=date.today().year - 10)))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(date.today()), max_value=date.today())
predict_days = st.sidebar.slider("🔮 Days to Predict", 1, 30, 7)
sequence_length = 60

# 📂 Load & Display Stock Data
df = load_stock_data("MSFT", start_date, end_date)
st.subheader("📂 Microsoft Stock Data")
st.dataframe(df.tail())

# 📊 Compute Technical Indicators
df = compute_rsi(df)
df = compute_macd(df)

st.subheader("📊 RSI & MACD Indicators")
st.pyplot(plot_rsi_macd(df))

# 🔁 Prepare LSTM Input
data = df[['Close']]
scaled_data, scaler = scale_data(data.values)
x, y = create_sequences(scaled_data, seq_len=sequence_length)

split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

x_train = x_train.reshape((-1, sequence_length, 1))
x_test = x_test.reshape((-1, sequence_length, 1))

# 🧠 Build & Train LSTM Model
model = build_model((x_train.shape[1], 1))
with st.spinner("🔁 Training LSTM model on MSFT..."):
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)

# 📉 Predict & Display Actual vs Predicted
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

valid = df.iloc[-len(predictions):].copy()
valid['Predictions'] = predictions
train = df.iloc[:len(df) - len(predictions)].copy()

st.subheader("📉 Actual vs Predicted Prices")
st.pyplot(plot_predictions(train, valid))

# 🔮 Future Forecast
last_60 = scaled_data[-sequence_length:]
future_preds = predict_future(model, last_60, predict_days, scaler)
future_df = build_future_df(df['Date'].iloc[-1], future_preds)

st.subheader(f"🗓️ {predict_days}-Day Forecast for MSFT")
st.dataframe(future_df)

# 🎯 Forecast Summary
avg_price = round(future_df['Predicted'].mean(), 2)
high_price = round(future_df['Predicted'].max(), 2)
low_price = round(future_df['Predicted'].min(), 2)

st.markdown(f"""
### 🎯 Forecast Summary
- **Average Target Price**: ${avg_price}
- **High Target Price**: ${high_price}
- **Low Target Price**: ${low_price}
""")

# 📊 Forecast Price Range Bar
st.plotly_chart(plot_forecast_range_bar(low_price, avg_price, high_price), use_container_width=True)

# 📈 Combined Historical + Forecast Line
st.plotly_chart(plot_unified_forecast_line(df, future_df), use_container_width=True)

# 🧠 Analyst Sentiment (Mocked)
ratings = {'Buy': 15, 'Hold': 5, 'Sell': 2}
st.plotly_chart(plot_analyst_pie_chart(ratings), use_container_width=True)

# 📥 Download Forecast
st.download_button("⬇️ Download Forecast CSV",
                   data=future_df.to_csv(index=False).encode(),
                   file_name="MSFT_forecast.csv",
                   mime='text/csv')

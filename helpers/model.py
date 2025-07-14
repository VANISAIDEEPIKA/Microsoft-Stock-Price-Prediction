# ---------------- helpers/model.py ----------------
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

def scale_data(data):
    """Scale data using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def create_sequences(data, seq_len):
    """Create rolling window sequences for LSTM."""
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(x), np.array(y)

def build_model(input_shape):
    """Build and compile a BiLSTM model."""
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=False), input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future(model, last_seq, days, scaler):
    """
    Predict future prices using the last known sequence.
    Returns unscaled predictions.
    """
    predictions = []
    current_input = last_seq.copy()

    for _ in range(days):
        pred = model.predict(current_input.reshape(1, -1, 1), verbose=0)[0]
        predictions.append(pred[0])
        current_input = np.append(current_input[1:], pred)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()

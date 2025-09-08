import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ------------------------------
# Load Pre-trained Model
# ------------------------------
model = load_model("Stock Prediction Model1.keras")

# ------------------------------
# Streamlit App Title
# ------------------------------
st.title("📈 Stock Market Price Prediction")

# ------------------------------
# User Input
# ------------------------------
stock = st.text_input("Enter Stock Symbol (e.g. AAPL, GOOG, MSFT)", "GOOG")
start = "2012-01-01"
end = None

# ------------------------------
# Fetch Stock Data
# ------------------------------
data = yf.download(stock, start, end)

# Show stock data (Latest → Oldest)
data_reversed = data.iloc[::-1]
st.subheader(f"Full Stock Data for {stock}")
st.dataframe(data_reversed)   # shows full scrollable table

# ------------------------------
# Data Splitting (use chronological order for training/prediction)
# ------------------------------
data_train = pd.DataFrame(data["Close"][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data["Close"][int(len(data)*0.80):])

# ------------------------------
# Data Preprocessing
# ------------------------------
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# ------------------------------
# Moving Average 50 Days
# ------------------------------
st.subheader("Price vs MA50")
ma_50 = data["Close"].rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(data["Close"], label="Closing Price", color="g")
ax1.plot(ma_50, label="MA50", color="r")
ax1.legend()
st.pyplot(fig1)

# ------------------------------
# Moving Average 50 vs 100 Days
# ------------------------------
st.subheader("Price vs MA50 vs MA100")
ma_100 = data["Close"].rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(data["Close"], label="Closing Price", color="g")
ax2.plot(ma_50, label="MA50", color="r")
ax2.plot(ma_100, label="MA100", color="b")
ax2.legend()
st.pyplot(fig2)

# ------------------------------
# Moving Average 100 vs 200 Days
# ------------------------------
st.subheader("Price vs MA100 vs MA200")
ma_200 = data["Close"].rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.plot(data["Close"], label="Closing Price", color="g")
ax3.plot(ma_100, label="MA100", color="r")
ax3.plot(ma_200, label="MA200", color="b")
ax3.legend()
st.pyplot(fig3)

# ------------------------------
# Prepare Test Data for Prediction
# ------------------------------
x_test, y_test = [], []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# ------------------------------
# Prediction
# ------------------------------
y_predicted = model.predict(x_test)

# Inverse scaling
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# ------------------------------
# Plot Predictions vs Actual
# ------------------------------
st.subheader("Predicted vs Actual Stock Price")
fig4, ax4 = plt.subplots(figsize=(10,5))
ax4.plot(y_test, label="Actual Price", color="g")
ax4.plot(y_predicted, label="Predicted Price", color="r")
ax4.set_xlabel("Time")
ax4.set_ylabel("Price")
ax4.legend()
st.pyplot(fig4)

# ------------------------------
# Display Key Insights
# ------------------------------
st.subheader("📊 Key Insights")
st.markdown(f"""
- The model was trained on **{stock}** stock price data starting from {start}.
- Predictions are made using an **LSTM deep learning model**.
- Moving averages are shown separately for **50, 100, and 200 days**.
- The stock data is now displayed in **both chronological order** and **reverse order** for convenience.
- The graph above compares the **actual closing prices** vs the **predicted prices**.
""")

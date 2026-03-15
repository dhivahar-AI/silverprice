import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as ygo
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Silver Price Prediction", page_icon="🪙", layout="wide")

st.title("🪙 Silver Price (SLV) Prediction Dashboard")
st.markdown("This dashboard extracts the last 5 years of silver ETF (SLV) prices from Yahoo Finance and predicts the prices for the next 1 year (365 days) using a hybrid machine learning model (Linear Trend + Random Forest Seasonality).")

@st.cache_data(ttl=3600)
def load_data():
    # Download 5 years of Silver historical data
    ticker = "SLV"
    df = yf.download(ticker, period="5y")
    
    # Check if data is mult-level columns (yfinance sometimes returns this)
    if isinstance(df.columns, pd.MultiIndex):
        # We just want the 'Close' prices for SLV
        df = df['Close']
        df = pd.DataFrame(df)
        df.columns = ['Close'] # Flatten renaming
    else:
        df = df[['Close']]
    
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

with st.spinner('Downloading 5 years of Silver Data...'):
    data = load_data()

# Show tail of data
st.subheader("Recent Historical Data")
st.dataframe(data.tail())

# --- Model Preparation and Feature Engineering ---
# Add time-based features
data['Date'] = pd.to_datetime(data['Date'])
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
data['Month'] = data['Date'].dt.month
data['DayOfYear'] = data['Date'].dt.dayofyear
data['DayOfWeek'] = data['Date'].dt.dayofweek

# We need scalar 1D arrays for model fitting
X_trend = data[['Days']].values
y = data['Close'].values

# Fit a Linear Regression model for the long-term trend
lin_reg = LinearRegression()
lin_reg.fit(X_trend, y)
data['Trend'] = lin_reg.predict(X_trend)

# Calculate residuals (What's left over after trend)
data['Residuals'] = data['Close'] - data['Trend']

# Fit a Random Forest to the residuals based on seasonal/time features
X_season = data[['Month', 'DayOfYear', 'DayOfWeek']].values
y_residual = data['Residuals'].values

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_season, y_residual)
data['Predicted_Residuals'] = rf_reg.predict(X_season)

# Combine for final training prediction
data['Predicted_Close'] = data['Trend'] + data['Predicted_Residuals']

# Calculate training RMSE
rmse = np.sqrt(mean_squared_error(data['Close'], data['Predicted_Close']))

st.success(f"Model trained successfully! Training RMSE: **${rmse:.2f}**")

# --- Forecasting the Next 1 Year (365 Days) ---
last_date = data['Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 366)]
future_df = pd.DataFrame({'Date': future_dates})

# Feature engineering for future dates
future_df['Days'] = (future_df['Date'] - data['Date'].min()).dt.days
future_df['Month'] = future_df['Date'].dt.month
future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek

# Predict Trend for future
X_future_trend = future_df[['Days']].values
future_df['Trend'] = lin_reg.predict(X_future_trend)

# Predict Residuals for future
X_future_season = future_df[['Month', 'DayOfYear', 'DayOfWeek']].values
future_df['Predicted_Residuals'] = rf_reg.predict(X_future_season)

# Combine Future Prediction
future_df['Forecast_Close'] = future_df['Trend'] + future_df['Predicted_Residuals']

# Metrics Display
current_price = data['Close'].iloc[-1]
final_forecast_price = future_df['Forecast_Close'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Latest Historical Price", f"${current_price:.2f}")
col2.metric("Forecasted Price (in 1 Year)", f"${final_forecast_price:.2f}", 
            f"{(final_forecast_price - current_price):.2f}")
col3.metric("Model RMSE", f"${rmse:.2f}")

# --- Plotly Visualization ---
st.subheader("Silver Price: Historical & 1-Year Forecast")

fig = ygo.Figure()

# Plot historical actuals
fig.add_trace(ygo.Scatter(x=data['Date'], y=data['Close'], mode='lines', 
                          name='Historical Price ($)', line=dict(color='blue')))

# Plot forecasted
fig.add_trace(ygo.Scatter(x=future_df['Date'], y=future_df['Forecast_Close'], mode='lines', 
                          name='Forecasted Price ($)', line=dict(color='orange', dash='dash')))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### How the Model Works")
st.info("""
To provide a forecast that accounts for both the overall long-term direction of silver and its yearly cyclical patterns, we use a hybrid approach:
1. **Trend Estimation:** A Linear Regression model captures the overall macro trend over the last 5 years based purely on daily progression.
2. **Seasonality Capture:** We subtracted the trend from the actual price to find the 'residuals'. A Random Forest Regressor is then trained on these residuals using seasonal features (Month, Day of Year, Day of Week).
3. **Combination:** The final prediction is the sum of the predicted long-term trend and the predicted seasonal residual.
""")

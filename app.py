import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv', parse_dates=['InvoiceDate'])

# Streamlit App Setup
st.title('Demand Forecasting System')

# Dropdown for selecting stock code
stock_code = st.selectbox('Select Stock Code:', data['StockCode'].unique())

# Input for forecast weeks
weeks = st.slider('Select number of weeks to forecast:', 1, 15, 5)

# Initialize forecast variable
forecast = None

# Show forecast plot (using ARIMA model as an example)
if st.button('Forecast'):
    # Filter sales data for the selected stock code
    product_sales = data[data['StockCode'] == stock_code].set_index('InvoiceDate')['Quantity']
    
    # Ensure that there are enough data points to fit the model
    if len(product_sales) > 5:  # Check if there are enough data points
        arima_model = ARIMA(product_sales, order=(5, 1, 0))
        arima_fit = arima_model.fit()
        forecast = arima_fit.forecast(steps=weeks)
        
        # Plotting
        st.subheader(f'Sales Forecast for Stock Code: {stock_code}')
        st.line_chart(product_sales)

        # Create a forecast series with appropriate date index
        forecast_index = pd.date_range(start=product_sales.index[-1] + pd.Timedelta(days=1), periods=weeks)
        forecast_series = pd.Series(forecast, index=forecast_index)
        st.line_chart(forecast_series)
    else:
        st.error("Not enough data points to fit the ARIMA model.")

# Allow users to download CSV of the forecast
if forecast is not None:
    csv = forecast_series.to_csv()  # Use forecast_series for download
    st.download_button(label="Download Forecast as CSV", data=csv, file_name='forecast.csv')
else:
    st.warning("Click the 'Forecast' button to generate a forecast before downloading.")

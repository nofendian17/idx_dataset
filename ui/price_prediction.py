import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

from utils.data_loader import load_stock_data

def page_price_prediction(stock_codes: list, available_dates: list):
    """
    Renders the 'Stock Price Prediction' page.
    """
    st.title("Stock Price Prediction (Simple Linear Regression)")

    st.warning(
        """
        **Disclaimer:** This prediction is for educational purposes only and uses a very
        simple Linear Regression model based on historical prices. It does not account for
        market volatility, news, or other external factors. **Do not use this for
        actual financial decisions.**
        """
    )

    stock_code = st.selectbox("Select Stock for Prediction", stock_codes)
    days_to_predict = st.number_input("Number of days to predict", 1, 30, 5)

    if st.button("Predict Future Prices"):
        try:
            # Load all available data for the stock to train the model
            df = load_stock_data(stock_code, min(available_dates), max(available_dates))

            if len(df) < 30:
                st.error("Not enough historical data to make a reliable prediction.")
                return

            # --- Model Training ---
            df['Future Price'] = df['Close Price'].shift(-1)
            df.dropna(inplace=True)

            X = df[['Close Price']]
            y = df['Future Price']

            model = LinearRegression()
            model.fit(X, y)

            # --- Prediction ---
            future_predictions = []
            last_price = df['Close Price'].iloc[-1]

            for _ in range(days_to_predict):
                next_pred = model.predict(np.array([[last_price]]))[0]
                future_predictions.append(next_pred)
                last_price = next_pred

            # --- Display Results ---
            last_date = df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

            st.subheader(f"Predicted Prices for the next {days_to_predict} days")
            st.dataframe(pred_df)

            # --- Visualization ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close Price'], mode='lines', name='Historical Prices'))
            fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Price'], mode='lines', name='Predicted Prices', line=dict(dash='dot')))
            fig.update_layout(title=f"Price Prediction for {stock_code}", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

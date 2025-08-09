import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go

from utils.data_loader import load_stock_data

def page_price_prediction(stock_codes: list, available_dates: list):
    """
    Renders the 'Stock Price Prediction' page with multiple model options.
    """
    st.title("Stock Price Prediction")

    st.warning(
        """
        **Disclaimer:** This prediction is for educational purposes only. The models used
        are based on historical prices and do not account for market volatility, news,
        or other external factors. **Do not use this for actual financial decisions.**
        """
    )

    stock_code = st.selectbox("Select Stock for Prediction", stock_codes)
    days_to_predict = st.number_input("Number of days to predict", 1, 30, 5)
    model_choice = st.selectbox(
        "Select Prediction Model",
        ["Linear Regression", "Random Forest", "XGBoost", "LightGBM"]
    )

    if st.button("Predict Future Prices"):
        try:
            # Load all available data for the stock to train the model
            df = load_stock_data(stock_code, min(available_dates), max(available_dates))

            if len(df) < 30:
                st.error("Not enough historical data to make a reliable prediction.")
                return

            # --- Feature Engineering ---
            df = engineer_features(df)

            # --- Model Training ---
            df['Future Price'] = df['Close Price'].shift(-1)
            df.dropna(inplace=True)

            # Define features and target
            features = ['Close Price', '7_day_MA', '14_day_MA', 'momentum', 'volatility', 'volume_trend', 'log_volume']
            X = df[features]

            # Ensure Future Price is valid before log transform
            df['Future Price'] = df['Future Price'].clip(lower=0.0001)
            y = np.log1p(df['Future Price'])  # Log transform the target

            # Train the selected model
            model = train_model(model_choice, X, y)

            # --- Prediction ---
            future_predictions = []
            last_row = df.iloc[-1]

            for _ in range(days_to_predict):
                # Create feature vector for prediction
                X_pred = pd.DataFrame([last_row[features]], columns=features)
                next_pred_log = model.predict(X_pred)[0]
                next_pred = np.expm1(next_pred_log)  # Inverse transform the prediction

                future_predictions.append(next_pred)

                # Update features for next prediction
                last_row = update_features_for_prediction(last_row, next_pred, features)

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

def engineer_features(df):
    """
    Engineer additional features for better price prediction.
    """
    # Calculate moving averages
    df['7_day_MA'] = df['Close Price'].rolling(window=7).mean()
    df['14_day_MA'] = df['Close Price'].rolling(window=14).mean()

    # Calculate price momentum (difference between current and previous close)
    df['momentum'] = df['Close Price'].diff()

    # Calculate volatility (standard deviation of recent prices)
    df['volatility'] = df['Close Price'].rolling(window=7).std()

    # Calculate volume trend
    df['volume_trend'] = df['Volume'].diff()

    # Add log-transformed volume (with validation)
    # Replace any zero or negative volumes with a small positive value
    df['Volume'] = df['Volume'].clip(lower=0.0001)
    df['log_volume'] = np.log1p(df['Volume'])

    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_to_scale = ['Close Price', '7_day_MA', '14_day_MA', 'momentum', 'volatility', 'volume_trend', 'log_volume']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Fill NaN values that result from rolling calculations
    df.bfill(inplace=True)

    return df

def train_model(model_name, X, y):
    """
    Train and return the selected model with appropriate parameters.
    """
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            random_state=42,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=1
        )
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=200,
            random_state=42,
            max_depth=4,
            learning_rate=0.05,
            min_child_samples=5,
            num_leaves=20,
            boosting_type='gbdt',
            lambda_l1=0.1,
            lambda_l2=0.1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    try:
        model.fit(X, y)
    except Exception as e:
        # If training fails, return a simple model that just predicts the mean
        st.warning(f"Model training failed: {e}. Using fallback prediction.")
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor(strategy="mean")
        model.fit(X, y)

    return model

def update_features_for_prediction(last_row, next_pred, features):
    """
    Update feature values for the next prediction iteration.
    """
    # Create a copy to avoid modifying the original
    new_row = last_row.copy()

    # Update close price and dependent features
    new_row['Close Price'] = next_pred

    # Update moving averages (simple approach: use the new price in the calculation)
    if '7_day_MA' in features:
        # Simple approach: average of last 6 prices + new price
        new_row['7_day_MA'] = (new_row['7_day_MA'] * 6 + next_pred) / 7

    if '14_day_MA' in features:
        # Simple approach: average of last 13 prices + new price
        new_row['14_day_MA'] = (new_row['14_day_MA'] * 13 + next_pred) / 14

    # Update momentum (difference from previous close)
    if 'momentum' in features:
        new_row['momentum'] = next_pred - last_row['Close Price']

    # Update volatility (simplified: keep the same for prediction)
    # In a more sophisticated approach, we'd recalculate this

    # Update volume trend (simplified: keep the same for prediction)
    # In a more sophisticated approach, we'd model this

    return new_row

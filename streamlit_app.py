import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from stock_analyzer import analyze_stock_data # Import the analysis function

# --- Configuration ---
DATA_DIR = 'data'

# --- Helper Functions ---

@st.cache_data
def get_available_files():
    """Gets a sorted list of available data files."""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('stock_data_') and f.endswith('.csv')]
    return sorted(files, reverse=True)

@st.cache_data
def get_all_stock_codes(files):
    """Gets a set of all unique stock codes from the latest file."""
    if not files:
        return []
    latest_file = files[0]
    df = pd.read_csv(os.path.join(DATA_DIR, latest_file))
    return sorted(df['Stock Code'].unique())

def get_date_from_filename(file):
    """Extracts date from a filename string."""
    date_str = file.replace('stock_data_', '').replace('.csv', '')
    return datetime.strptime(date_str, '%Y-%m-%d').date()

# --- Data Loading Functions ---

def load_daily_data(selected_date):
    """Loads stock data for a single selected day."""
    date_str = selected_date.strftime('%Y-%m-%d')
    file_path = os.path.join(DATA_DIR, f'stock_data_{date_str}.csv')
    if not os.path.exists(file_path):
        st.warning(f"No data available for {date_str}.")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    for col in ['Previous Price', 'Last Price', 'Volume', 'Value']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Previous Price', 'Last Price'], inplace=True)
    df = df[df['Previous Price'] > 0]
    df.rename(columns={'Last Price': 'Close Price'}, inplace=True) # Rename here for daily data
    return df

def load_stock_data(stock_code, start_date, end_date, files):
    """Loads historical data for a single stock code."""
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    filtered_files = [f for f in files if start_date_str <= f.replace('stock_data_', '').replace('.csv', '') <= end_date_str]
    if not filtered_files:
        raise ValueError(f"No data files found for date range {start_date_str} to {end_date_str}")

    dfs = [pd.read_csv(os.path.join(DATA_DIR, file)) for file in filtered_files]
    combined_df = pd.concat(dfs)
    df_stock = combined_df[combined_df['Stock Code'] == stock_code].copy()

    if df_stock.empty:
        raise ValueError(f"No data found for stock code {stock_code}")

    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock.sort_values('Date', inplace=True)
    df_stock.reset_index(drop=True, inplace=True)
    
    df_stock.rename(columns={'Last Price': 'Close Price'}, inplace=True)
    for col in ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume']:
        df_stock[col] = pd.to_numeric(df_stock[col], errors='coerce')
    # Removed df_stock.dropna(inplace=True) as it's handled in stock_analyzer.py
    return df_stock

# --- Strategy & Backtesting Functions ---

def calculate_backtest_metrics(df):
    """Calculates and returns key backtesting metrics."""
    # Ensure required columns exist
    if not {'buy_signal', 'sell_signal', 'Close Price'}.issubset(df.columns):
        return 0, 0, 0

    # Vectorized position tracking
    df['position'] = np.where(df['buy_signal'].notna(), 1, np.where(df['sell_signal'].notna(), -1, 0))
    df['position'] = df['position'].replace(0, np.nan).ffill().fillna(0)
    
    # Calculate trades
    df['trade_entry'] = df['buy_signal'].where(df['position'].diff() == 1)
    df['trade_exit'] = df['sell_signal'].where(df['position'].diff() == -1)
    
    # Handle unclosed positions (close at last price)
    if df['position'].iloc[-1] == 1:
        df.loc[df.index[-1], 'trade_exit'] = df['Close Price'].iloc[-1]
    
    # Track trades through position changes
    trades = []
    entry_price = None
    for i in range(len(df)):
        # Detect new entries (position change from 0 to 1)
        if df['position'].iloc[i] == 1 and (i == 0 or df['position'].iloc[i-1] != 1):
            entry_price = df['trade_entry'].iloc[i]
        # Detect exits (position change from 1 to 0)
        elif entry_price is not None and df['position'].iloc[i] != 1:
            exit_price = df['trade_exit'].iloc[i]
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price
            })
            entry_price = None

    # Handle any remaining open position
    if entry_price is not None:
        exit_price = df['trade_exit'].iloc[-1]
        trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price
        })
    
    trades = pd.DataFrame(trades)
    
    if trades.empty:
        return 0, 0, 0

    trades_df = trades.copy()
    # Calculate profit percentage with inf handling
    trades_df['profit_pct'] = (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price'].replace(0, np.nan) * 100
    trades_df['profit_pct'] = trades_df['profit_pct'].fillna(0).replace([np.inf, -np.inf], 0)
    
    total_return_pct = trades_df['profit_pct'].sum()
    win_rate = (trades_df['profit_pct'] > 0).mean() * 100 if not trades_df.empty else 0
    
    # Drawdown calculation remains the same as it's based on price, not trades.
    # Ensure 'Close Price' is available and valid.
    if 'Close Price' not in df.columns or df['Close Price'].isnull().all():
        max_drawdown = 0
    else:
        # Ensure Close Price is numeric and handle potential NaNs if any slipped through
        df['Close Price'] = pd.to_numeric(df['Close Price'], errors='coerce')
        df.dropna(subset=['Close Price'], inplace=True)
        
        if df['Close Price'].empty:
            max_drawdown = 0
        else:
            # Calculate cumulative returns with NaN and inf handling
            returns = df['Close Price'].pct_change().fillna(0)
            returns.replace([np.inf, -np.inf], 0, inplace=True)
            df['cumulative_return'] = (1 + returns).cumprod()
            df['running_max'] = df['cumulative_return'].cummax()
            df['drawdown'] = (df['running_max'] - df['cumulative_return']) / df['running_max']
            max_drawdown = df['drawdown'].max() * 100 if not df.empty else 0

    return total_return_pct, win_rate, max_drawdown

def sma_crossover_strategy(df, short_window=20, long_window=50):
    df['SMA_Short'] = talib.SMA(df['Close Price'], timeperiod=short_window)
    df['SMA_Long'] = talib.SMA(df['Close Price'], timeperiod=long_window)
    df['signal'] = 0
    df.loc[short_window:, 'signal'] = np.where(df['SMA_Short'][short_window:] > df['SMA_Long'][short_window:], 1, 0)
    df['position'] = df['signal'].diff()
    df['buy_signal'] = np.where(df['position'] == 1, df['Close Price'], np.nan)
    df['sell_signal'] = np.where(df['position'] == -1, df['Close Price'], np.nan)
    return df

def rsi_strategy(df, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=rsi_period)
    df['buy_signal'] = np.where((df['RSI'] < rsi_oversold) & (df['RSI'].shift(1) >= rsi_oversold), df['Close Price'], np.nan)
    df['sell_signal'] = np.where((df['RSI'] > rsi_overbought) & (df['RSI'].shift(1) <= rsi_overbought), df['Close Price'], np.nan)
    return df

def sma_rsi_strategy(df, short_window=20, long_window=50, rsi_period=14, rsi_buy_threshold=50, rsi_sell_threshold=50):
    df['SMA_Short'] = talib.SMA(df['Close Price'], timeperiod=short_window)
    df['SMA_Long'] = talib.SMA(df['Close Price'], timeperiod=long_window)
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=rsi_period)
    df['sma_signal'] = 0
    df.loc[long_window:, 'sma_signal'] = np.where(df['SMA_Short'][long_window:] > df['SMA_Long'][long_window:], 1, -1)
    df['buy_signal'] = np.where((df['sma_signal'] == 1) & (df['sma_signal'].shift(1) == -1) & (df['RSI'] > rsi_buy_threshold), df['Close Price'], np.nan)
    df['sell_signal'] = np.where((df['sma_signal'] == -1) & (df['sma_signal'].shift(1) == 1) & (df['RSI'] < rsi_sell_threshold), df['Close Price'], np.nan)
    return df

def swing_trading_strategy(df):
    df['SMA50'] = talib.SMA(df['Close Price'], timeperiod=50)
    df['SMA20'] = talib.SMA(df['Close Price'], timeperiod=20)
    df['RSI14'] = talib.RSI(df['Close Price'], timeperiod=14)
    
    df['buy_signal'] = np.nan
    df['sell_signal'] = np.nan
    
    in_position = False
    for i in range(1, len(df)):
        # Entry Condition
        if not in_position:
            if (df['Close Price'][i] > df['SMA50'][i] and 
                df['RSI14'][i] < 50 and 
                df['RSI14'][i] > df['RSI14'][i-1] and
                df['Close Price'][i] > df['Open Price'][i]): # Green candle
                df.loc[i, 'buy_signal'] = df['Close Price'][i]
                in_position = True
        # Exit Condition
        else:
            if df['RSI14'][i] > 70 or df['Close Price'][i] < df['SMA20'][i]:
                df.loc[i, 'sell_signal'] = df['Close Price'][i]
                in_position = False
    return df

def day_trading_ema_rsi_strategy(df, short_ema=9, long_ema=21, rsi_period=7, rsi_overbought=80, rsi_oversold=20):
    """Day trading strategy based on EMA crossover and RSI filter."""
    df['EMA_Short'] = talib.EMA(df['Close Price'], timeperiod=short_ema)
    df['EMA_Long'] = talib.EMA(df['Close Price'], timeperiod=long_ema)
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=rsi_period)
    
    df['buy_signal'] = np.nan
    df['sell_signal'] = np.nan
    
    position = 0 
    for i in range(1, len(df)):
        # Buy Condition
        if df['EMA_Short'][i] > df['EMA_Long'][i] and df['EMA_Short'][i-1] <= df['EMA_Long'][i-1] and df['RSI'][i] < rsi_overbought:
            if position != 1:
                df.loc[i, 'buy_signal'] = df['Close Price'][i]
                position = 1
        # Sell Condition
        elif df['EMA_Short'][i] < df['EMA_Long'][i] and df['EMA_Short'][i-1] >= df['EMA_Long'][i-1] and df['RSI'][i] > rsi_oversold:
            if position == 1:
                df.loc[i, 'sell_signal'] = df['Close Price'][i]
                position = 0
    return df

def ema_rsi_tp_sl_strategy(df, short_ema=5, long_ema=9, rsi_period=7, rsi_threshold=50, take_profit_pct=0.03, stop_loss_pct=0.02):
    """
    Strategy based on EMA crossover, RSI filter, and Take Profit/Stop Loss.
    Buy: EMA5 > EMA9 and RSI(7) > 50
    Sell: EMA5 < EMA9 or Stop Loss or Take Profit hit
    """
    df['EMA_Short'] = talib.EMA(df['Close Price'], timeperiod=short_ema)
    df['EMA_Long'] = talib.EMA(df['Close Price'], timeperiod=long_ema)
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=rsi_period)
    
    df['buy_signal'] = np.nan
    df['sell_signal'] = np.nan
    
    in_position = False
    buy_price = 0
    
    for i in range(1, len(df)):
        # Buy Condition
        if not in_position:
            if df['EMA_Short'][i] > df['EMA_Long'][i] and df['RSI'][i] > rsi_threshold:
                df.loc[i, 'buy_signal'] = df['Close Price'][i]
                buy_price = df['Close Price'][i]
                in_position = True
        # Sell Condition
        else:
            current_price = df['Close Price'][i]
            # Check for Stop Loss or Take Profit
            if (current_price <= buy_price * (1 - stop_loss_pct)) or \
               (current_price >= buy_price * (1 + take_profit_pct)):
                df.loc[i, 'sell_signal'] = current_price
                in_position = False
                buy_price = 0
            # Check for EMA crossover sell signal
            elif df['EMA_Short'][i] < df['EMA_Long'][i]:
                df.loc[i, 'sell_signal'] = current_price
                in_position = False
                buy_price = 0
    return df

# --- NEW STRATEGY FUNCTION ---
def day_trading_strategy_user(df, short_ema_period=5, long_ema_period=20, rsi_period=7, rsi_buy_threshold=50, rsi_sell_threshold=50, tp_pct=0.02, sl_pct=0.01, volume_lookback=5):
    """
    Day trading strategy based on EMA crossover, RSI filter, VWAP, Volume, and TP/SL.
    Buy: EMA5 cross up EMA20 + Price > VWAP + RSI(7) > 50 + Volume increase
    Sell: EMA5 cross down EMA20 or Price < VWAP or TP/SL hit
    """
    df['EMA_Short'] = talib.EMA(df['Close Price'], timeperiod=short_ema_period)
    df['EMA_Long'] = talib.EMA(df['Close Price'], timeperiod=long_ema_period)
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=rsi_period)
    
    # Calculate VWAP
    # Ensure High, Low, Close Price are numeric and handle potential NaNs
    df['High Price'] = pd.to_numeric(df['High Price'], errors='coerce')
    df['Low Price'] = pd.to_numeric(df['Low Price'], errors='coerce')
    df['Close Price'] = pd.to_numeric(df['Close Price'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Drop rows where essential price/volume data might be missing after coercion
    df.dropna(subset=['High Price', 'Low Price', 'Close Price', 'Volume'], inplace=True)
    
    df['Typical Price'] = (df['High Price'] + df['Low Price'] + df['Close Price']) / 3
    # Calculate cumulative sums for VWAP
    df['Cumulative_TP_Volume'] = (df['Typical Price'] * df['Volume']).cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    # Avoid division by zero if Cumulative_Volume is 0
    df['VWAP'] = np.where(df['Cumulative_Volume'] > 0, df['Cumulative_TP_Volume'] / df['Cumulative_Volume'], 0)
    
    # Calculate average volume
    df['Avg_Volume'] = df['Volume'].rolling(window=volume_lookback).mean()
    
    df['buy_signal'] = np.nan
    df['sell_signal'] = np.nan
    
    in_position = False
    buy_price = 0
    
    for i in range(1, len(df)):
        # Buy Condition
        if not in_position:
            # EMA5 cross up EMA20
            ema_cross_up = df['EMA_Short'][i] > df['EMA_Long'][i] and df['EMA_Short'][i-1] <= df['EMA_Long'][i-1]
            # Price above VWAP
            price_above_vwap = df['Close Price'][i] > df['VWAP'][i]
            # RSI > 50
            rsi_condition = df['RSI'][i] > rsi_buy_threshold
            # Volume increased
            volume_increase = df['Volume'][i] > df['Avg_Volume'][i]
            
            if ema_cross_up and price_above_vwap and rsi_condition and volume_increase:
                df.loc[i, 'buy_signal'] = df['Close Price'][i] # This is where the buy signal is set
                buy_price = df['Close Price'][i]
                in_position = True
        # Sell Condition
        else:
            current_price = df['Close Price'][i]
            
            # EMA5 cross down EMA20
            ema_cross_down = df['EMA_Short'][i] < df['EMA_Long'][i] and df['EMA_Short'][i-1] >= df['EMA_Long'][i-1]
            # Price below VWAP
            price_below_vwap = df['Close Price'][i] < df['VWAP'][i]
            
            # Check for Take Profit or Stop Loss
            tp_hit = current_price >= buy_price * (1 + tp_pct)
            sl_hit = current_price <= buy_price * (1 - sl_pct)
            
            if ema_cross_down or price_below_vwap or tp_hit or sl_hit:
                df.loc[i, 'sell_signal'] = current_price # This is where the sell signal is set
                in_position = False
                buy_price = 0
                
    return df

# --- Visualization Functions ---
def create_trend_visualization(df, stock_code, sma_period=20):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2], specs=[[{"type": "candlestick"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "bar"}]])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open Price'], high=df['High Price'], low=df['Low Price'], close=df['Close Price'], name='Candlestick'), row=1, col=1)
    df[f'SMA_{sma_period}'] = talib.SMA(df['Close Price'], timeperiod=sma_period)
    df['EMA_12'] = talib.EMA(df['Close Price'], timeperiod=12)
    df['EMA_26'] = talib.EMA(df['Close Price'], timeperiod=26)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[f'SMA_{sma_period}'], mode='lines', name=f'SMA {sma_period}', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_12'], mode='lines', name='EMA 12', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_26'], mode='lines', name='EMA 26', line=dict(width=1)), row=1, col=1)
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=14)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    macd, macdsignal, macdhist = talib.MACD(df['Close Price'], fastperiod=12, slowperiod=26, signalperiod=9)
    fig.add_trace(go.Scatter(x=df['Date'], y=macd, mode='lines', name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=macdsignal, mode='lines', name='MACD Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df['Date'], y=macdhist, name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=4, col=1)
    fig.update_layout(title=f"Stock Analysis: {stock_code}", height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    return fig

def display_market_summary(df):
    st.header(f"Market Summary for {df['Date'].iloc[0]}")
    df['Change'] = df['Close Price'] - df['Previous Price']
    df['% Change'] = (df['Change'] / df['Previous Price']) * 100
    st.subheader("Top 10 Gainers")
    gainers = df.sort_values('% Change', ascending=False).head(10)
    st.dataframe(gainers[['Stock Code', 'Close Price', 'Change', '% Change']])
    st.subheader("Top 10 Losers")
    losers = df.sort_values('% Change', ascending=True).head(10)
    st.dataframe(losers[['Stock Code', 'Close Price', 'Change', '% Change']])
    st.subheader("Top 10 Most Active by Volume")
    active_volume = df.sort_values('Volume', ascending=False).head(10)
    st.dataframe(active_volume[['Stock Code', 'Volume', 'Close Price', '% Change']])
    st.subheader("Top 10 Most Active by Value")
    active_value = df.sort_values('Value', ascending=False).head(10)
    st.dataframe(active_value[['Stock Code', 'Value', 'Close Price', '% Change']])

# --- Streamlit App Pages ---

def page_trend_analysis(stock_codes, files):
    st.header("Stock Trend Analysis")
    stock_code = st.sidebar.selectbox("Select Stock", stock_codes)
    available_dates = [get_date_from_filename(f) for f in files]
    min_date, max_date = min(available_dates), max(available_dates)
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    sma_period = st.sidebar.number_input("SMA Period", min_value=1, max_value=100, value=20)
    try:
        df = load_stock_data(stock_code, start_date, end_date, files)
        fig = create_trend_visualization(df, stock_code, sma_period) 
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

def page_market_summary(files):
    st.header("Daily Market Summary")
    available_dates = [get_date_from_filename(f) for f in files]
    selected_date = st.date_input("Select Date", max(available_dates), min_value=min(available_dates), max_value=max(available_dates))
    df = load_daily_data(selected_date)
    if not df.empty:
        display_market_summary(df)

def page_price_prediction(stock_codes, files):
    st.header("Stock Price Prediction (Simple Linear Regression)")
    stock_code = st.selectbox("Select Stock for Prediction", stock_codes)
    days_to_predict = st.number_input("Number of days to predict", 1, 30, 5)
    if st.button("Predict Future Prices"):
        try:
            available_dates = [get_date_from_filename(f) for f in files]
            df = load_stock_data(stock_code, min(available_dates), max(available_dates), files)
            if len(df) < 30:
                st.error("Not enough data for prediction.")
                return
            df['Target'] = df['Close Price'].shift(-1)
            df.dropna(inplace=True)
            X = df[['Close Price']]
            y = df['Target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Model Mean Squared Error (MSE): {mse:.2f}")
            future_predictions = []
            last_price = df['Close Price'].iloc[-1]
            for _ in range(days_to_predict):
                last_price_df = pd.DataFrame([[last_price]], columns=['Close Price'])
                next_pred = model.predict(last_price_df)[0]
                future_predictions.append(next_pred)
                last_price = next_pred
            last_date = df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
            st.subheader(f"Predicted Prices for the next {days_to_predict} days")
            st.dataframe(pred_df)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close Price'], mode='lines', name='Historical Prices'))
            fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Price'], mode='lines', name='Predicted Prices', line=dict(dash='dot')))
            fig.update_layout(title=f"Price Prediction for {stock_code}", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def page_strategy_analysis(stock_codes, files):
    st.header("Strategy Analysis & Backtesting")
    stock_code = st.selectbox("Select Stock", stock_codes, key='strategy_stock')
    
    strategy_options = [
        "SMA Crossover", 
        "RSI Strategy", 
        "SMA Crossover + RSI Filter", 
        "Swing Trading (SMA50 + RSI Pullback)", 
        "Day Trading (EMA9/21 Cross + RSI7)", 
        "EMA5/9 Cross + RSI7 + TP/SL",
        "Day Trading (EMA5/20 Cross + RSI7 + VWAP + Vol + TP/SL)" # New strategy
    ]
    strategy = st.selectbox("Select Strategy", strategy_options)
    
    available_dates = [get_date_from_filename(f) for f in files]
    min_date, max_date = min(available_dates), max(available_dates)
    
    default_start_date = max_date - timedelta(days=30)
    if default_start_date < min_date:
        default_start_date = min_date

    start_date = st.date_input("Start Date", default_start_date, min_value=min_date, max_value=max_date, key='strat_start')
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key='strat_end')
    
    if st.button("Analyze Strategy"):
        try:
            df = load_stock_data(stock_code, start_date, end_date, files)
            
            if strategy == "SMA Crossover":
                df = sma_crossover_strategy(df)
                st.info("Buy when short SMA crosses above long SMA. Sell when it crosses below.")
            elif strategy == "RSI Strategy":
                df = rsi_strategy(df)
                st.info("Buy when RSI becomes > 30 (from oversold). Sell when RSI becomes < 70 (from overbought).")
            elif strategy == "SMA Crossover + RSI Filter":
                df = sma_rsi_strategy(df)
                st.info("Buy on SMA golden cross if RSI > 50. Sell on SMA death cross if RSI < 50.")
            elif strategy == "Swing Trading (SMA50 + RSI Pullback)":
                df = swing_trading_strategy(df)
                st.info("Buy on uptrend (Price > SMA50) during an RSI pullback. Sell when overbought (RSI > 70) or trend weakens (Price < SMA20).")
            elif strategy == "Day Trading (EMA9/21 Cross + RSI7)":
                df = day_trading_ema_rsi_strategy(df)
                st.info("Buy when EMA9 crosses above EMA21 (and RSI < 80). Sell when EMA9 crosses below EMA21 (and RSI > 20). This is a simulation on daily data.")
            elif strategy == "EMA5/9 Cross + RSI7 + TP/SL":
                df = ema_rsi_tp_sl_strategy(df)
                st.info("Buy when EMA5 crosses above EMA9 and RSI(7) > 50. Sell on EMA5 < EMA9, or TP (+3%), or SL (-2%).")
            elif strategy == "Day Trading (EMA5/20 Cross + RSI7 + VWAP + Vol + TP/SL)":
                df = day_trading_strategy_user(df, short_ema_period=5, long_ema_period=20, rsi_period=7, rsi_buy_threshold=50, rsi_sell_threshold=50, tp_pct=0.02, sl_pct=0.01, volume_lookback=5)
                st.info("Strategy: EMA5 crosses above EMA20, Price > VWAP, RSI(7) > 50, Volume > Avg Volume (5 days). Sell on EMA5 < EMA20, Price < VWAP, or TP/SL hit.")

            total_return, win_rate, max_drawdown = calculate_backtest_metrics(df.copy())
            st.subheader("Backtest Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return (%)", f"{total_return:.2f}%")
            col2.metric("Win Rate (%)", f"{win_rate:.2f}%")
            col3.metric("Max Drawdown (%)", f"{max_drawdown:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close Price'], mode='lines', name='Price'))
            
            # Plotting indicators based on the selected strategy
            if strategy == "SMA Crossover":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Short'], mode='lines', name='SMA Short', line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Long'], mode='lines', name='SMA Long', line=dict(width=1, dash='dot')))
            elif strategy == "RSI Strategy":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', yaxis='y2'))
                fig.update_layout(yaxis2=dict(title='RSI', overlaying='y', side='right'))
            elif strategy == "SMA Crossover + RSI Filter":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Short'], mode='lines', name='SMA Short', line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Long'], mode='lines', name='SMA Long', line=dict(width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', yaxis='y2'))
                fig.update_layout(yaxis2=dict(title='RSI', overlaying='y', side='right'))
            elif strategy == "Swing Trading (SMA50 + RSI Pullback)":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], mode='lines', name='SMA50', line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], mode='lines', name='SMA20', line=dict(width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI14'], mode='lines', name='RSI14', yaxis='y2'))
                fig.update_layout(yaxis2=dict(title='RSI', overlaying='y', side='right'))
            elif strategy == "Day Trading (EMA9/21 Cross + RSI7)":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_Short'], mode='lines', name='EMA 9', line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_Long'], mode='lines', name='EMA 21', line=dict(width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI 7', yaxis='y2'))
                fig.update_layout(yaxis2=dict(title='RSI', overlaying='y', side='right'))
            elif strategy == "EMA5/9 Cross + RSI7 + TP/SL":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_Short'], mode='lines', name='EMA 5', line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_Long'], mode='lines', name='EMA 9', line=dict(width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI 7', yaxis='y2'))
                fig.update_layout(yaxis2=dict(title='RSI', overlaying='y', side='right'))
            # --- Plotting for the new strategy ---
            elif strategy == "Day Trading (EMA5/20 Cross + RSI7 + VWAP + Vol + TP/SL)":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_Short'], mode='lines', name='EMA 5', line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_Long'], mode='lines', name='EMA 20', line=dict(width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI 7', yaxis='y2'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['VWAP'], mode='lines', name='VWAP', line=dict(width=1, dash='dash')))
                # Plotting volume as a bar chart on a third y-axis
                fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', yaxis='y3', marker_color='rgba(0,0,0,0.1)'))
                
                fig.update_layout(
                    title=f"{strategy} Analysis for {stock_code}", 
                    xaxis_title="Date", 
                    yaxis_title="Price",
                    yaxis2=dict(title='RSI 7', overlaying='y', side='right', position=0.9), # Position for RSI
                    yaxis3=dict(title='Volume', overlaying='y', side='right', position=0.95, showgrid=False), # Position for Volume
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
            else:
                # Default layout for other strategies if they plot RSI
                if 'RSI' in df.columns and strategy in ["RSI Strategy", "SMA Crossover + RSI Filter", "Swing Trading (SMA50 + RSI Pullback)", "Day Trading (EMA9/21 Cross + RSI7)", "EMA5/9 Cross + RSI7 + TP/SL"]:
                    fig.update_layout(
                        title=f"{strategy} Analysis for {stock_code}", 
                        xaxis_title="Date", 
                        yaxis_title="Price",
                        yaxis2=dict(title='RSI', overlaying='y', side='right'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                else:
                    fig.update_layout(
                        title=f"{strategy} Analysis for {stock_code}", 
                        xaxis_title="Date", 
                        yaxis_title="Price",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
            
            # Plot Buy and Sell Signals
            # Create separate columns for plotting signals to ensure only signal points are plotted
            df['plot_buy_signal'] = np.where(df['buy_signal'].notna(), df['Close Price'], np.nan)
            df['plot_sell_signal'] = np.where(df['sell_signal'].notna(), df['Close Price'], np.nan)

            fig.add_trace(go.Scatter(x=df['Date'], y=df['plot_buy_signal'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=10)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['plot_sell_signal'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=10)))
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Main App ---
def main():
    st.set_page_config(layout="wide")
    st.title("IDX Stock Analysis Dashboard")
    files = get_available_files()
    if not files:
        st.error("No data files found in 'data' directory.")
        return
    stock_codes = get_all_stock_codes(files)
    st.sidebar.title("Navigation")
    page_options = ["Daily Market Summary", "Stock Trend Analysis", "Strategy Analysis", "Price Prediction", "Comprehensive Trend Analysis"]
    page = st.sidebar.radio("Choose a page", page_options)
    if page == "Daily Market Summary":
        page_market_summary(files)
    elif page == "Stock Trend Analysis":
        page_trend_analysis(stock_codes, files)
    elif page == "Strategy Analysis":
        page_strategy_analysis(stock_codes, files)
    elif page == "Price Prediction":
        page_price_prediction(stock_codes, files)
    elif page == "Comprehensive Trend Analysis":
        page_comprehensive_trend_analysis()

def page_comprehensive_trend_analysis():
    st.header("Comprehensive Stock Trend Analysis")
    st.write("This section provides a comprehensive trend analysis for all available stocks based on SMA, MACD, and ADX indicators.")

    if st.button("Run Comprehensive Analysis"):
        with st.spinner("Analyzing stock data... This may take a while."):
            analysis_results = analyze_stock_data(DATA_DIR)
            if not analysis_results.empty:
                st.subheader("Latest Stock Analysis Results")
                
                # Add filters
                st.sidebar.subheader("Filter Analysis Results")
                
                # Get unique values for filters, add "All" option
                all_trends = ["All"] + list(analysis_results['Trend'].unique())
                all_strengths = ["All"] + list(analysis_results['Strength'].unique())
                all_phases = ["All"] + list(analysis_results['Phase'].unique())
                all_signals = ["All"] + list(analysis_results['Signal'].unique())

                selected_trend = st.sidebar.selectbox("Filter by Trend", all_trends)
                selected_strength = st.sidebar.selectbox("Filter by Strength", all_strengths)
                selected_phase = st.sidebar.selectbox("Filter by Phase", all_phases)
                selected_signal = st.sidebar.selectbox("Filter by Signal", all_signals)

                filtered_results = analysis_results.copy()

                if selected_trend != "All":
                    filtered_results = filtered_results[filtered_results['Trend'] == selected_trend]
                if selected_strength != "All":
                    filtered_results = filtered_results[filtered_results['Strength'] == selected_strength]
                if selected_phase != "All":
                    filtered_results = filtered_results[filtered_results['Phase'] == selected_phase]
                if selected_signal != "All":
                    filtered_results = filtered_results[filtered_results['Signal'] == selected_signal]

                st.dataframe(filtered_results)
                
                # Option to download results
                csv = filtered_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Results as CSV",
                    data=csv,
                    file_name="filtered_stock_analysis.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No analysis results generated. Please ensure data files are available.")

if __name__ == "__main__":
    main()

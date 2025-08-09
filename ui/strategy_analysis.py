import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import talib
from datetime import datetime, timedelta

from utils.data_loader import load_stock_data
from utils.strategies import (
    sma_crossover_strategy, rsi_strategy, sma_rsi_strategy,
    swing_trading_strategy, day_trading_ema_rsi_strategy,
    ema_rsi_tp_sl_strategy, day_trading_strategy_user
)
from utils.backtesting import calculate_backtest_metrics

def get_strategy_function(strategy_name: str):
    """Returns the function corresponding to the selected strategy."""
    strategies = {
        "SMA Crossover": sma_crossover_strategy,
        "RSI Strategy": rsi_strategy,
        "SMA Crossover + RSI Filter": sma_rsi_strategy,
        "Swing Trading (SMA50 + RSI Pullback)": swing_trading_strategy,
        "Day Trading (EMA9/21 Cross + RSI7)": day_trading_ema_rsi_strategy,
        "EMA5/9 Cross + RSI7 + TP/SL": ema_rsi_tp_sl_strategy,
        "Day Trading (EMA5/20 Cross + RSI7 + VWAP + Vol + TP/SL)": day_trading_strategy_user,
    }
    return strategies.get(strategy_name)

def plot_strategy(df: pd.DataFrame, stock_code: str, strategy: str) -> go.Figure:
    """
    Plots the price chart with buy/sell signals and relevant indicators for a strategy.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close Price'], mode='lines', name='Price'))

    # Plot indicators based on strategy
    # This part can be expanded to show relevant indicators for each strategy
    if 'SMA_Short' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Short'], mode='lines', name='SMA Short', line=dict(width=1)))
    if 'SMA_Long' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Long'], mode='lines', name='SMA Long', line=dict(width=1, dash='dot')))
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', yaxis='y2'))

    # Plot signals
    buy_signals = df[df['buy_signal'].notna()]
    sell_signals = df[df['sell_signal'].notna()]

    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close Price'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=10)))
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close Price'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=10)))

    fig.update_layout(
        title=f"{strategy} Analysis for {stock_code}",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(title='RSI', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def page_strategy_analysis(stock_codes: list, available_dates: list):
    """
    Renders the 'Strategy Analysis & Backtesting' page.
    """
    st.title("Strategy Analysis & Backtesting")

    stock_code = st.selectbox("Select Stock", stock_codes, key='strategy_stock')

    strategy_options = [
        "SMA Crossover", "RSI Strategy", "SMA Crossover + RSI Filter",
        "Swing Trading (SMA50 + RSI Pullback)", "Day Trading (EMA9/21 Cross + RSI7)",
        "EMA5/9 Cross + RSI7 + TP/SL", "Day Trading (EMA5/20 Cross + RSI7 + VWAP + Vol + TP/SL)"
    ]
    strategy_name = st.selectbox("Select Strategy", strategy_options)

    min_date, max_date = min(available_dates), max(available_dates)
    default_start = max_date - timedelta(days=90)
    start_date = st.date_input("Start Date", default_start if default_start > min_date else min_date, min_value=min_date, max_value=max_date, key='strat_start')
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key='strat_end')

    if st.button("Analyze Strategy"):
        try:
            df = load_stock_data(stock_code, start_date, end_date)
            if df.empty:
                st.warning(f"No data for {stock_code} in the selected date range.")
                return

            strategy_func = get_strategy_function(strategy_name)
            if not strategy_func:
                st.error("Selected strategy is not implemented.")
                return

            df_strategy = strategy_func(df.copy())

            total_return, win_rate, max_drawdown = calculate_backtest_metrics(df_strategy.copy())

            st.subheader("Backtest Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return (%)", f"{total_return:.2f}%")
            col2.metric("Win Rate (%)", f"{win_rate:.2f}%")
            col3.metric("Max Drawdown (%)", f"{max_drawdown:.2f}%")

            fig = plot_strategy(df_strategy, stock_code, strategy_name)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during strategy analysis: {e}")

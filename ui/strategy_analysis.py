import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from typing import Dict, Callable, Optional

from utils.data_loader import load_stock_data
from utils.strategies import (
    sma_crossover_strategy, rsi_strategy, sma_rsi_strategy,
    swing_trading_strategy, day_trading_ema_rsi_strategy,
    ema_rsi_tp_sl_strategy, day_trading_strategy_user,
    support_resistance_strategy
)
from utils.backtesting import calculate_backtest_metrics

# Constants
STRATEGY_OPTIONS = {
    "SMA Crossover": sma_crossover_strategy,
    "RSI Strategy": rsi_strategy,
    "SMA Crossover + RSI Filter": sma_rsi_strategy,
    "Swing Trading (SMA50 + RSI Pullback)": swing_trading_strategy,
    "Day Trading (EMA9/21 Cross + RSI7)": day_trading_ema_rsi_strategy,
    "EMA5/9 Cross + RSI7 + TP/SL": ema_rsi_tp_sl_strategy,
    "Day Trading (EMA5/20 Cross + RSI7 + VWAP + Vol + TP/SL)": day_trading_strategy_user,
    "Support & Resistance": support_resistance_strategy,
    }

INDICATOR_CONFIG = {
    'MA5': {'color': 'red', 'width': 1},
    'MA20': {'color': 'green', 'width': 1},
    'MA60': {'color': 'blue', 'width': 1},
    'SMA_Short': {'color': 'orange', 'width': 1},
    'SMA_Long': {'color': 'purple', 'width': 1, 'dash': 'dot'},
    'EMA_Short': {'color': 'blue', 'width': 1},
    'EMA_Long': {'color': 'orange', 'width': 1, 'dash': 'dot'},
    'RSI': {'color': 'gray', 'width': 1, 'yaxis': 'y2'},
    'ADX': {'color': 'purple', 'width': 1, 'yaxis': 'y3'},
    'ATR': {'color': 'brown', 'width': 1, 'yaxis': 'y4'},
}

DATE_RANGE_PRESETS = {
    "Last 90 Days": 90,
    "Last 180 Days": 180,
    "Last 365 Days": 365,
    "All Data": None
}

MIN_DATA_POINTS = 30

def get_strategy_function(strategy_name: str) -> Optional[Callable]:
    """Returns the function corresponding to the selected strategy."""
    return STRATEGY_OPTIONS.get(strategy_name)

def create_base_chart(df: pd.DataFrame) -> go.Figure:
    """Creates the base candlestick chart."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open Price'],
        high=df['High Price'],
        low=df['Low Price'],
        close=df['Close Price'],
        name='Candlesticks'
    ))
    return fig

def add_indicators(fig: go.Figure, df: pd.DataFrame) -> Dict[str, dict]:
    """Adds indicators to the chart and returns layout updates."""
    layout_updates = {}
    
    # Add moving averages
    for ma_col in ['MA5', 'MA20', 'MA60']:
        if ma_col in df.columns:
            config = INDICATOR_CONFIG[ma_col]
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[ma_col],
                mode='lines',
                name=ma_col,
                line=dict(width=config['width'], color=config['color'])
            ))

    # Add strategy-specific indicators
    for indicator, config in INDICATOR_CONFIG.items():
        if indicator in df.columns and indicator not in ['MA5', 'MA20', 'MA60']:
            line_config = {k: v for k, v in config.items() 
                         if k in ['width', 'color', 'dash']}
            
            trace = go.Scatter(
                x=df['Date'],
                y=df[indicator],
                mode='lines',
                name=indicator,
                line=line_config
            )

            if 'yaxis' in config:
                axis_name = config['yaxis']  # e.g. 'y2'
                trace['yaxis'] = axis_name
                
                # Create proper Plotly axis reference (e.g. 'yaxis2' instead of 'y2')
                proper_axis_name = f"yaxis{axis_name[1:]}"  # Convert 'y2' to 'yaxis2'
                
                if proper_axis_name not in layout_updates:
                    axis_title = indicator if indicator == 'RSI' else indicator.upper()
                    position = 0.95 if axis_name == 'y3' else 0.05
                    side = 'right' if axis_name in ['y2', 'y3'] else 'left'
                    
                    layout_updates[proper_axis_name] = {
                        'title': axis_title,
                        'overlaying': 'y',
                        'side': side,
                        'position': position,
                        'showgrid': False
                    }

            fig.add_trace(trace)
    
    return layout_updates

def add_signals(fig: go.Figure, df: pd.DataFrame) -> None:
    """Adds buy/sell signals to the chart."""
    buy_signals = df[df['buy_signal'].notna()]
    sell_signals = df[df['sell_signal'].notna()]

    fig.add_trace(go.Scatter(
        x=buy_signals['Date'],
        y=buy_signals['buy_signal'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', symbol='triangle-up', size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=sell_signals['Date'],
        y=sell_signals['sell_signal'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', symbol='triangle-down', size=10)
    ))

def plot_strategy(df: pd.DataFrame, stock_code: str, strategy: str) -> go.Figure:
    """Plots the price chart with indicators and signals."""
    fig = create_base_chart(df)
    layout_updates = add_indicators(fig, df)
    add_signals(fig, df)

    # Base layout
    layout = {
        'title': f"{strategy} Analysis for {stock_code}",
        'xaxis': {'title': "Date"},
        'yaxis': {'title': "Price"},
        'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    }

    # Add secondary axes configuration
    if layout_updates:
        layout.update(layout_updates)

    fig.update_layout(layout)
    return fig

def display_metrics(metrics: Dict[str, float]) -> None:
    """Displays metrics in an organized layout."""
    metrics_layout = {
        "Performance": [
            ("Total Return (%)", f"{metrics['total_return_pct']:.2f}%"),
            ("Number of Trades", f"{metrics['num_trades']}"),
            ("Avg Trade (%)", f"{metrics['avg_trade_pct']:.2f}%")
        ],
        "Risk Management": [
            ("Max Drawdown (%)", f"{metrics['max_drawdown_pct']:.2f}%"),
            ("Win Rate (%)", f"{metrics['win_rate_pct']:.2f}%"),
            ("Profit Factor", f"{metrics['profit_factor']:.2f}")
        ],
        "Risk-Adjusted": [
            ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
            ("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
        ]
    }

    for category, category_metrics in metrics_layout.items():
        st.subheader(category)
        cols = st.columns(len(category_metrics))
        for i, (metric_name, metric_value) in enumerate(category_metrics):
            cols[i].metric(metric_name, metric_value)

def validate_inputs(stock_code: str, strategy_name: str, 
                  start_date, end_date, df: pd.DataFrame) -> bool:
    """Validates all inputs before analysis."""
    if not stock_code:
        st.error("Please select a stock code.")
        return False
    
    if not strategy_name:
        st.error("Please select a strategy.")
        return False
    
    if start_date > end_date:
        st.error("Invalid date range. Start date must be before end date.")
        return False
    
    if df.empty:
        st.warning(f"No data for {stock_code} in the selected date range.")
        return False
    
    if len(df) < MIN_DATA_POINTS:
        st.warning(f"Insufficient data points ({len(df)}) for reliable analysis. Minimum {MIN_DATA_POINTS} required.")
        return False
    
    return True

def get_date_range(date_range_option: str, min_date, max_date):
    """Returns appropriate start date based on preset."""
    if date_range_option == "Custom":
        return min_date
    
    if date_range_option == "All Data":
        return min_date
    
    days = DATE_RANGE_PRESETS[date_range_option]
    default_start = max_date - timedelta(days=days)
    return default_start if default_start > min_date else min_date

def page_strategy_analysis(stock_codes: list, available_dates: list):
    """Renders the 'Strategy Analysis & Backtesting' page."""
    st.title("Strategy Analysis & Backtesting")

    # Input validation
    if not stock_codes:
        st.warning("No stock codes available. Please ensure data is loaded.")
        return

    if not available_dates:
        st.warning("No available dates. Please ensure data is loaded.")
        return

    # User inputs
    stock_code = st.selectbox("Select Stock", stock_codes, key='strategy_stock')
    strategy_name = st.selectbox("Select Strategy", list(STRATEGY_OPTIONS.keys()))
    
    min_date, max_date = min(available_dates), max(available_dates)
    
    # Date range selection
    date_range_option = st.radio(
        "Select Date Range Preset:",
        ["Custom"] + list(DATE_RANGE_PRESETS.keys()),
        horizontal=True
    )
    
    # Set default start date
    default_start = get_date_range(date_range_option, min_date, max_date)
    
    start_date = st.date_input(
        "Start Date", 
        default_start, 
        min_value=min_date, 
        max_value=max_date, 
        key='strat_start'
    )
    
    end_date = st.date_input(
        "End Date", 
        max_date, 
        min_value=min_date, 
        max_value=max_date, 
        key='strat_end'
    )

    if st.button("Analyze Strategy"):
        with st.spinner("Running analysis..."):
            try:
                # Load data
                df = load_stock_data(stock_code, start_date, end_date)
                
                # Validate inputs
                if not validate_inputs(stock_code, strategy_name, start_date, end_date, df):
                    return
                
                # Get strategy function
                strategy_func = get_strategy_function(strategy_name)
                if not strategy_func:
                    st.error("Selected strategy is not implemented.")
                    return
                
                # Apply strategy
                df_strategy = strategy_func(df.copy())
                
                # Validate signals
                if 'buy_signal' not in df_strategy.columns or 'sell_signal' not in df_strategy.columns:
                    st.error("Strategy did not generate valid signals.")
                    return
                
                if df_strategy['buy_signal'].isna().all() and df_strategy['sell_signal'].isna().all():
                    st.warning("No buy/sell signals generated.")
                    return
                
                # Calculate metrics
                metrics = calculate_backtest_metrics(df_strategy.copy())
                
                # Display results
                st.subheader("Backtest Results")
                display_metrics(metrics)
                
                # Plot results
                fig = plot_strategy(df_strategy, stock_code, strategy_name)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred during strategy analysis: {str(e)}")
                st.exception(e)

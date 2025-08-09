import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
from datetime import datetime

from utils.data_loader import load_stock_data

def create_trend_visualization(df, stock_code, sma_period=20):
    """
    Creates a comprehensive trend visualization with price, indicators, and volume.
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        specs=[[{"type": "candlestick"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "bar"}]]
    )

    # Row 1: Candlestick and SMAs/EMAs
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open Price'], high=df['High Price'], low=df['Low Price'], close=df['Close Price'], name='Candlestick'), row=1, col=1)
    df[f'SMA_{sma_period}'] = talib.SMA(df['Close Price'], timeperiod=sma_period)
    df['EMA_12'] = talib.EMA(df['Close Price'], timeperiod=12)
    df['EMA_26'] = talib.EMA(df['Close Price'], timeperiod=26)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[f'SMA_{sma_period}'], mode='lines', name=f'SMA {sma_period}', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_12'], mode='lines', name='EMA 12', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_26'], mode='lines', name='EMA 26', line=dict(width=1)), row=1, col=1)

    # Row 2: RSI
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=14)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # Row 3: MACD
    macd, macdsignal, macdhist = talib.MACD(df['Close Price'], fastperiod=12, slowperiod=26, signalperiod=9)
    fig.add_trace(go.Scatter(x=df['Date'], y=macd, mode='lines', name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=macdsignal, mode='lines', name='MACD Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df['Date'], y=macdhist, name='MACD Hist'), row=3, col=1)

    # Row 4: Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=4, col=1)

    # Layout Updates
    fig.update_layout(
        title=f"Stock Analysis: {stock_code}",
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

    return fig

def page_trend_analysis(stock_codes: list, available_dates: list):
    """
    Renders the 'Stock Trend Analysis' page.
    """
    st.title("Stock Trend Analysis")

    stock_code = st.sidebar.selectbox("Select Stock", stock_codes)

    min_date, max_date = min(available_dates), max(available_dates)
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    sma_period = st.sidebar.number_input("SMA Period", min_value=1, max_value=100, value=20)

    try:
        df = load_stock_data(stock_code, start_date, end_date)
        if df.empty:
            st.warning(f"No data available for {stock_code} in the selected date range.")
            return

        fig = create_trend_visualization(df, stock_code, sma_period)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

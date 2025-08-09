import streamlit as st
import pandas as pd
from datetime import datetime

from utils.data_loader import load_daily_data

def display_market_summary(df: pd.DataFrame, selected_date: datetime.date):
    """
    Displays the market summary, including top gainers, losers, and most active stocks.
    """
    st.header(f"Market Summary for {selected_date.strftime('%Y-%m-%d')}")

    # Calculate change and percentage change
    df['Change'] = df['Close Price'] - df['Previous Price']
    df['% Change'] = (df['Change'] / df['Previous Price']) * 100

    # --- Display Metrics ---
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

def page_market_summary(available_dates: list):
    """
    Renders the 'Daily Market Summary' page.
    """
    st.title("Daily Market Summary")

    selected_date = st.date_input(
        "Select Date",
        max(available_dates),
        min_value=min(available_dates),
        max_value=max(available_dates)
    )

    df = load_daily_data(selected_date)

    if not df.empty:
        display_market_summary(df, selected_date)
    else:
        st.warning(f"No data available for {selected_date.strftime('%Y-%m-%d')}.")

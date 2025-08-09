import os
import pandas as pd
import streamlit as st
from datetime import datetime

DATA_DIR = 'data'

@st.cache_data
def get_available_files():
    """Gets a sorted list of available data files."""
    if not os.path.exists(DATA_DIR):
        return []
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('stock_data_') and f.endswith('.csv')]
    return sorted(files, reverse=True)

@st.cache_data
def get_all_stock_codes():
    """Gets a set of all unique stock codes from the latest file."""
    files = get_available_files()
    if not files:
        return []
    try:
        latest_file = files[0]
        df = pd.read_csv(os.path.join(DATA_DIR, latest_file))
        return sorted(df['Stock Code'].unique())
    except Exception:
        return []

def get_date_from_filename(file):
    """Extracts date from a filename string."""
    date_str = file.replace('stock_data_', '').replace('.csv', '')
    return datetime.strptime(date_str, '%Y-%m-%d').date()

@st.cache_data
def load_daily_data(selected_date):
    """Loads and prepares stock data for a single day."""
    date_str = selected_date.strftime('%Y-%m-%d')
    file_path = os.path.join(DATA_DIR, f'stock_data_{date_str}.csv')

    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df.rename(columns={'Last Price': 'Close Price'}, inplace=True)

    numeric_cols = ['Previous Price', 'Close Price', 'Volume', 'Value']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Previous Price', 'Close Price'], inplace=True)
    df = df[df['Previous Price'] > 0]

    return df

@st.cache_data
def load_stock_data(stock_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Loads, filters, and prepares historical data for a single stock code."""
    files = get_available_files()
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Filter files to the relevant date range first
    relevant_files = [
        f for f in files
        if start_date_str <= get_date_from_filename(f).strftime('%Y-%m-%d') <= end_date_str
    ]

    if not relevant_files:
        return pd.DataFrame()

    df_list = [pd.read_csv(os.path.join(DATA_DIR, file)) for file in relevant_files]

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    stock_df = combined_df[combined_df['Stock Code'] == stock_code].copy()

    if stock_df.empty:
        return pd.DataFrame()

    # Data preparation
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df.sort_values('Date', inplace=True)
    stock_df.reset_index(drop=True, inplace=True)
    stock_df.rename(columns={'Last Price': 'Close Price'}, inplace=True)

    numeric_cols = ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume']
    for col in numeric_cols:
        stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')

    stock_df.dropna(subset=numeric_cols, inplace=True)

    return stock_df

import pandas as pd
import numpy as np
import os
import talib
import logging
import sys
from typing import List, Optional

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- Data Loading and Preparation ---

def load_and_prepare_data(data_dir: str = 'data') -> Optional[pd.DataFrame]:
    """
    Loads all stock data from CSV files in a directory, and prepares it for analysis.
    """
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')])

    if not all_files:
        logging.warning("No data files found in the directory.")
        return None

    logging.info(f"Loading {len(all_files)} files...")
    df_list = []
    for filepath in all_files:
        try:
            df_temp = pd.read_csv(filepath)
            df_list.append(df_temp)
        except Exception as e:
            logging.error(f"Error reading {filepath}: {e}")

    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)
    logging.info("Data loaded. Preparing for analysis...")

    # --- Data Cleaning ---
    df.rename(columns={'Last Price': 'Close Price'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    price_cols = ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume']
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=price_cols, inplace=True)
    df.sort_values(by=['Stock Code', 'Date'], inplace=True)

    return df

# --- Indicator Calculation ---

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for the stock data.
    """
    logging.info("Calculating technical indicators...")

    df = df.copy()
    grouped = df.groupby('Stock Code')

    df['SMA50'] = grouped['Close Price'].transform(lambda x: talib.SMA(x, 50))
    df['SMA200'] = grouped['Close Price'].transform(lambda x: talib.SMA(x, 200))

    def calculate_macd_adx(x):
        macd_line, signal_line, _ = talib.MACD(x['Close Price'])
        adx = talib.ADX(x['High Price'], x['Low Price'], x['Close Price'], 14)
        return pd.DataFrame({
            'MACD_line': macd_line,
            'Signal_line': signal_line,
            'ADX': adx
        }, index=x.index)

    indicator_df = grouped.apply(calculate_macd_adx)

    # Reset the 'Stock Code' level of the index to align for the join
    indicator_df = indicator_df.reset_index(level=0, drop=True)

    df = df.join(indicator_df)

    df['SMA50_slope'] = df.groupby('Stock Code')['SMA50'].diff()

    return df

# --- Trend and Strength Analysis ---

def determine_trend_and_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the trend and its strength based on indicators.
    """
    logging.info("Determining trend and strength...")

    df['Trend'] = np.select(
        [
            (df['Close Price'] > df['SMA50']) & (df['SMA50'] > df['SMA200']) & (df['SMA50_slope'] > 0),
            (df['Close Price'] < df['SMA50']) & (df['SMA50'] < df['SMA200']) & (df['SMA50_slope'] < 0)
        ],
        ['Uptrend', 'Downtrend'],
        default='Sideways'
    )

    df['Strength'] = np.select(
        [df['ADX'] > 25, (df['ADX'] >= 20) & (df['ADX'] <= 25)],
        ['Strong', 'Moderate'],
        default='Weak'
    )

    return df

# --- Phase and Signal Generation ---

def identify_market_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the market phase (e.g., Accumulation, Advancing).
    """
    logging.info("Identifying market phase...")

    grouped = df.groupby('Stock Code')
    df['Prev_Trend'] = grouped['Trend'].shift(1)
    df['Prev_MACD_line'] = grouped['MACD_line'].shift(1)
    df['Prev_Signal_line'] = grouped['Signal_line'].shift(1)

    is_advancing = (df['Trend'] == 'Uptrend') & (df['Strength'] == 'Strong')
    is_declining = (df['Trend'] == 'Downtrend') & (df['Strength'] == 'Strong')
    is_accumulation = (df['Prev_Trend'].isin(['Downtrend', 'Sideways'])) & (df['Trend'] == 'Uptrend') & (df['MACD_line'] > df['Signal_line']) & (df['Prev_MACD_line'] <= df['Prev_Signal_line'])
    is_distribution = (df['Prev_Trend'] == 'Uptrend') & (df['Trend'].isin(['Downtrend', 'Sideways'])) & (df['MACD_line'] < df['Signal_line']) & (df['Prev_MACD_line'] >= df['Prev_Signal_line'])

    df['Phase'] = np.select(
        [is_advancing, is_declining, is_accumulation, is_distribution, df['Trend'] == 'Sideways'],
        ['Advancing', 'Declining', 'Accumulation', 'Distribution', 'Sideways'],
        default='Unknown'
    )
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates Buy/Sell/Hold signals based on trend, phase, and momentum.
    """
    logging.info("Generating signals...")

    bullish_momentum = (df['MACD_line'] > df['Signal_line']) & (df['MACD_line'] > 0)
    bearish_momentum = (df['MACD_line'] < df['Signal_line']) & (df['MACD_line'] < 0)

    is_buy = ((df['Phase'] == 'Accumulation') & (df['Strength'].isin(['Moderate', 'Strong']))) | ((df['Phase'] == 'Advancing') & bullish_momentum)
    is_sell = ((df['Phase'] == 'Distribution') & (df['Strength'].isin(['Moderate', 'Strong']))) | ((df['Phase'] == 'Declining') & bearish_momentum)

    df['Signal'] = np.select(
        [is_buy, is_sell, df['Trend'] == 'Sideways', df['Trend'] == 'Downtrend'],
        ['Buy', 'Sell', 'Neutral', 'Avoid'],
        default='Hold'
    )
    return df

# --- Main Analysis Function ---

def analyze_stock_data(data_dir: str = 'data') -> Optional[pd.DataFrame]:
    """
    Main function to run the full stock analysis pipeline.
    """
    df = load_and_prepare_data(data_dir)
    if df is None or df.empty:
        return pd.DataFrame()

    df = calculate_indicators(df)
    df = determine_trend_and_strength(df)
    df = identify_market_phase(df)
    df = generate_signals(df)

    logging.info("Analysis complete. Preparing final output.")
    latest_data = df.loc[df.groupby('Stock Code')['Date'].idxmax()]

    output_cols = ['Date', 'Stock Code', 'Trend', 'Strength', 'Phase', 'Signal']
    return latest_data[output_cols].copy()

# --- Execution ---

if __name__ == "__main__":
    analysis_results = analyze_stock_data()

    if not analysis_results.empty:
        logging.info("Stock Analysis Results (Latest Data per Stock):")
        print(analysis_results.to_string())
    else:
        logging.warning("No data was processed or analyzed.")

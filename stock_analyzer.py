import pandas as pd
import numpy as np
import os
import talib # Import talib for indicator calculations

def calculate_sma(data, window):
    return talib.SMA(data['Close Price'], timeperiod=window)

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    macd_line, signal_line, _ = talib.MACD(data['Close Price'],
                                            fastperiod=fast_period,
                                            slowperiod=slow_period,
                                            signalperiod=signal_period)
    return pd.DataFrame({'MACD_line': macd_line, 'Signal_line': signal_line})

def calculate_adx(data, window=14):
    adx = talib.ADX(data['High Price'], data['Low Price'], data['Close Price'], timeperiod=window)
    return adx

def analyze_stock_data(data_dir='data'):
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_files.sort() # Ensure consistent order

    df_list = []
    for filename in all_files:
        filepath = os.path.join(data_dir, filename)
        try:
            df_temp = pd.read_csv(filepath)
            df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    # Data Cleaning and Preparation
    df['Date'] = pd.to_datetime(df['Date'])
    # Renamed 'Last Price' to 'Close Price' for consistency with indicator calculations
    df.rename(columns={'Last Price': 'Close Price'}, inplace=True)
    
    # Ensure numeric types for price and volume columns before calculations
    for col in ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in essential columns for indicator calculation
    df.dropna(subset=['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume'], inplace=True)
    
    df.sort_values(by=['Stock Code', 'Date'], inplace=True)

    # Calculate Indicators
    df['SMA50'] = df.groupby('Stock Code', group_keys=False).apply(lambda x: calculate_sma(x, 50))
    df['SMA200'] = df.groupby('Stock Code', group_keys=False).apply(lambda x: calculate_sma(x, 200))

    macd_results = df.groupby('Stock Code', group_keys=False).apply(lambda x: calculate_macd(x))
    df['MACD_line'] = macd_results['MACD_line']
    df['Signal_line'] = macd_results['Signal_line']

    df['ADX'] = df.groupby('Stock Code', group_keys=False).apply(lambda x: calculate_adx(x))

    # Calculate SMA50 slope
    df['SMA50_slope'] = df.groupby('Stock Code')['SMA50'].diff()

    # Initialize trend columns
    df['Trend'] = 'Sideways'
    df['Strength'] = 'Weak'
    df['Signal'] = 'Hold'
    df['Phase'] = 'Unknown' # New column for trend phase

    # Apply trend rules
    uptrend_condition = (df['Close Price'] > df['SMA50']) & (df['SMA50'] > df['SMA200']) & (df['SMA50_slope'] > 0)
    downtrend_condition = (df['Close Price'] < df['SMA50']) & (df['SMA50'] < df['SMA200']) & (df['SMA50_slope'] < 0)

    df.loc[uptrend_condition, 'Trend'] = 'Uptrend'
    df.loc[downtrend_condition, 'Trend'] = 'Downtrend'

    # Apply strength rules based on ADX
    strong_trend_condition = df['ADX'] > 25
    weak_trend_condition = df['ADX'] < 20
    moderate_trend_condition = (df['ADX'] >= 20) & (df['ADX'] <= 25)

    df.loc[strong_trend_condition, 'Strength'] = 'Strong'
    df.loc[weak_trend_condition, 'Strength'] = 'Weak'
    df.loc[moderate_trend_condition, 'Strength'] = 'Moderate'

    # MACD Momentum
    df['MACD_bullish_momentum'] = (df['MACD_line'] > df['Signal_line']) & (df['MACD_line'] > 0)
    df['MACD_bearish_momentum'] = (df['MACD_line'] < df['Signal_line']) & (df['MACD_line'] < 0)

    # Trend Phase Detection (requires looking at previous day's trend)
    # Sort by Stock Code and Date to ensure correct shifting
    df.sort_values(by=['Stock Code', 'Date'], inplace=True)
    df['Prev_Trend'] = df.groupby('Stock Code')['Trend'].shift(1)
    df['Prev_Strength'] = df.groupby('Stock Code')['Strength'].shift(1)
    df['Prev_MACD_line'] = df.groupby('Stock Code')['MACD_line'].shift(1)
    df['Prev_Signal_line'] = df.groupby('Stock Code')['Signal_line'].shift(1)

    # Advancing: Strong Uptrend
    df.loc[(df['Trend'] == 'Uptrend') & (df['Strength'] == 'Strong'), 'Phase'] = 'Advancing'

    # Declining: Strong Downtrend
    df.loc[(df['Trend'] == 'Downtrend') & (df['Strength'] == 'Strong'), 'Phase'] = 'Declining'

    # Accumulation: Transition from Downtrend/Sideways to Uptrend, with bullish signs
    # Accumulation: Transition from Downtrend/Sideways to Uptrend, with bullish signs
    # Refined: Was not Uptrend, now is Uptrend, and MACD cross up (can be below or above zero)
    accumulation_condition = (df['Prev_Trend'].isin(['Downtrend', 'Sideways'])) & \
                             (df['Trend'] == 'Uptrend') & \
                             (df['MACD_line'] > df['Signal_line']) & \
                             (df['Prev_MACD_line'] <= df['Prev_Signal_line'])
    df.loc[accumulation_condition, 'Phase'] = 'Accumulation'

    # Distribution: Transition from Uptrend to Downtrend/Sideways, with bearish signs
    # Refined: Was Uptrend, now is not Uptrend, and MACD cross down (can be below or above zero)
    distribution_condition = (df['Prev_Trend'] == 'Uptrend') & \
                             (df['Trend'].isin(['Downtrend', 'Sideways'])) & \
                             (df['MACD_line'] < df['Signal_line']) & \
                             (df['Prev_MACD_line'] >= df['Prev_Signal_line'])
    df.loc[distribution_condition, 'Phase'] = 'Distribution'

    # If not explicitly set to Advancing, Declining, Accumulation, or Distribution,
    # and if the trend is Sideways, set phase to Sideways.
    df.loc[(df['Phase'] == 'Unknown') & (df['Trend'] == 'Sideways'), 'Phase'] = 'Sideways'
    # For other 'Unknown' phases, if it's an Uptrend but not strong, or Downtrend but not strong,
    # it might be a weaker Advancing/Declining or a continuation of Accumulation/Distribution.
    # For simplicity, let's keep it 'Unknown' if it doesn't fit the defined phases.

    # Refine Signal rules based on new Phases and MACD momentum
    # Initialize all signals to 'Hold'
    df['Signal'] = 'Hold'

    # Buy: Accumulation or Advancing with bullish MACD momentum, considering strength
    df.loc[((df['Phase'] == 'Accumulation') & (df['Strength'].isin(['Moderate', 'Strong']))) | \
           ((df['Phase'] == 'Advancing') & df['MACD_bullish_momentum']), 'Signal'] = 'Buy'

    # Sell: Distribution or Declining with bearish MACD momentum, considering strength
    df.loc[((df['Phase'] == 'Distribution') & (df['Strength'].isin(['Moderate', 'Strong']))) | \
           ((df['Phase'] == 'Declining') & df['MACD_bearish_momentum']), 'Signal'] = 'Sell'

    # Neutral: Sideways trend
    df.loc[(df['Trend'] == 'Sideways') & (df['Signal'] == 'Hold'), 'Signal'] = 'Neutral'

    # Avoid: Downtrend (unless already a 'Sell' signal)
    df.loc[(df['Trend'] == 'Downtrend') & (df['Signal'] == 'Hold'), 'Signal'] = 'Avoid'

    # Final Output Table
    latest_data = df.loc[df.groupby('Stock Code')['Date'].idxmax()]

    output_table = latest_data[['Date', 'Stock Code', 'Trend', 'Strength', 'Phase', 'Signal']].copy() # Added 'Phase'

    # Clean up intermediate columns if necessary
    # df.drop(columns=['Close Price', 'SMA50', 'SMA200', 'MACD_line', 'Signal_line', 'ADX', 'SMA50_slope'], inplace=True, errors='ignore')

    return output_table

if __name__ == "__main__":
    # Example usage:
    # Assuming the script is run from the root directory where 'data' folder is located
    analysis_results = analyze_stock_data()

    if not analysis_results.empty:
        print("Stock Analysis Results (Latest Data per Stock):")
        print(analysis_results.to_string()) # Use to_string to display all rows and columns
    else:
        print("No data processed.")

    # To save the results to a CSV file:
    # if not analysis_results.empty:
    #     analysis_results.to_csv('stock_analysis_output.csv', index=False)
    #     print("\nResults saved to stock_analysis_output.csv")

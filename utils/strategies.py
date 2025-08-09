import pandas as pd
import numpy as np
import talib

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
    df['buy_signal'] = np.where((df['sma_signal'] == 1) & (df['sma_signal'].shift(1) == -1) & (df['RSI'] > rsi_buy_.threshold), df['Close Price'], np.nan)
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
        if not in_position:
            if (df['Close Price'][i] > df['SMA50'][i] and
                df['RSI14'][i] < 50 and
                df['RSI14'][i] > df['RSI14'][i-1] and
                df['Close Price'][i] > df['Open Price'][i]):
                df.loc[i, 'buy_signal'] = df['Close Price'][i]
                in_position = True
        else:
            if df['RSI14'][i] > 70 or df['Close Price'][i] < df['SMA20'][i]:
                df.loc[i, 'sell_signal'] = df['Close Price'][i]
                in_position = False
    return df

def day_trading_ema_rsi_strategy(df, short_ema=9, long_ema=21, rsi_period=7, rsi_overbought=80, rsi_oversold=20):
    df['EMA_Short'] = talib.EMA(df['Close Price'], timeperiod=short_ema)
    df['EMA_Long'] = talib.EMA(df['Close Price'], timeperiod=long_ema)
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=rsi_period)

    df['buy_signal'] = np.nan
    df['sell_signal'] = np.nan

    position = 0
    for i in range(1, len(df)):
        if df['EMA_Short'][i] > df['EMA_Long'][i] and df['EMA_Short'][i-1] <= df['EMA_Long'][i-1] and df['RSI'][i] < rsi_overbought:
            if position != 1:
                df.loc[i, 'buy_signal'] = df['Close Price'][i]
                position = 1
        elif df['EMA_Short'][i] < df['EMA_Long'][i] and df['EMA_Short'][i-1] >= df['EMA_Long'][i-1] and df['RSI'][i] > rsi_oversold:
            if position == 1:
                df.loc[i, 'sell_signal'] = df['Close Price'][i]
                position = 0
    return df

def ema_rsi_tp_sl_strategy(df, short_ema=5, long_ema=9, rsi_period=7, rsi_threshold=50, take_profit_pct=0.03, stop_loss_pct=0.02):
    df['EMA_Short'] = talib.EMA(df['Close Price'], timeperiod=short_ema)
    df['EMA_Long'] = talib.EMA(df['Close Price'], timeperiod=long_ema)
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=rsi_period)

    df['buy_signal'] = np.nan
    df['sell_signal'] = np.nan

    in_position = False
    buy_price = 0

    for i in range(1, len(df)):
        if not in_position:
            if df['EMA_Short'][i] > df['EMA_Long'][i] and df['RSI'][i] > rsi_threshold:
                df.loc[i, 'buy_signal'] = df['Close Price'][i]
                buy_price = df['Close Price'][i]
                in_position = True
        else:
            current_price = df['Close Price'][i]
            if (current_price <= buy_price * (1 - stop_loss_pct)) or \
               (current_price >= buy_price * (1 + take_profit_pct)):
                df.loc[i, 'sell_signal'] = current_price
                in_position = False
                buy_price = 0
            elif df['EMA_Short'][i] < df['EMA_Long'][i]:
                df.loc[i, 'sell_signal'] = current_price
                in_position = False
                buy_price = 0
    return df

def day_trading_strategy_user(df, short_ema_period=5, long_ema_period=20, rsi_period=7, rsi_buy_threshold=50, rsi_sell_threshold=50, tp_pct=0.02, sl_pct=0.01, volume_lookback=5):
    df['EMA_Short'] = talib.EMA(df['Close Price'], timeperiod=short_ema_period)
    df['EMA_Long'] = talib.EMA(df['Close Price'], timeperiod=long_ema_period)
    df['RSI'] = talib.RSI(df['Close Price'], timeperiod=rsi_period)

    df['Typical Price'] = (df['High Price'] + df['Low Price'] + df['Close Price']) / 3
    df['Cumulative_TP_Volume'] = (df['Typical Price'] * df['Volume']).cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = np.where(df['Cumulative_Volume'] > 0, df['Cumulative_TP_Volume'] / df['Cumulative_Volume'], 0)

    df['Avg_Volume'] = df['Volume'].rolling(window=volume_lookback).mean()

    df['buy_signal'] = np.nan
    df['sell_signal'] = np.nan

    in_position = False
    buy_price = 0

    for i in range(1, len(df)):
        if not in_position:
            ema_cross_up = df['EMA_Short'][i] > df['EMA_Long'][i] and df['EMA_Short'][i-1] <= df['EMA_Long'][i-1]
            price_above_vwap = df['Close Price'][i] > df['VWAP'][i]
            rsi_condition = df['RSI'][i] > rsi_buy_threshold
            volume_increase = df['Volume'][i] > df['Avg_Volume'][i]

            if ema_cross_up and price_above_vwap and rsi_condition and volume_increase:
                df.loc[i, 'buy_signal'] = df['Close Price'][i]
                buy_price = df['Close Price'][i]
                in_position = True
        else:
            current_price = df['Close Price'][i]

            ema_cross_down = df['EMA_Short'][i] < df['EMA_Long'][i] and df['EMA_Short'][i-1] >= df['EMA_Long'][i-1]
            price_below_vwap = df['Close Price'][i] < df['VWAP'][i]

            tp_hit = current_price >= buy_price * (1 + tp_pct)
            sl_hit = current_price <= buy_price * (1 - sl_pct)

            if ema_cross_down or price_below_vwap or tp_hit or sl_hit:
                df.loc[i, 'sell_signal'] = current_price
                in_position = False
                buy_price = 0

    return df

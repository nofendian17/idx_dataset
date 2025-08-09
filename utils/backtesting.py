import pandas as pd
import numpy as np

def calculate_backtest_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Calculates and returns key backtesting metrics: total return, win rate, and max drawdown.
    """
    if not {'buy_signal', 'sell_signal', 'Close Price'}.issubset(df.columns):
        return 0, 0, 0

    # --- Trade Calculation ---
    df['position'] = np.where(df['buy_signal'].notna(), 1, np.nan)
    df['position'] = np.where(df['sell_signal'].notna(), 0, df['position'])
    df['position'].ffill(inplace=True)
    df['position'].fillna(0, inplace=True)

    df['trade'] = df['position'].diff()

    trades = []
    entry_price = None
    entry_date = None

    for i, row in df.iterrows():
        if row['trade'] == 1: # Entry
            entry_price = row['buy_signal']
            entry_date = row['Date']
        elif row['trade'] == -1 and entry_price is not None: # Exit
            exit_price = row['sell_signal']
            exit_date = row['Date']
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
            })
            entry_price = None # Reset for next trade

    # If the last position is still open, close it at the last price
    if entry_price is not None:
        trades.append({
            'entry_date': entry_date,
            'exit_date': df.iloc[-1]['Date'],
            'entry_price': entry_price,
            'exit_price': df.iloc[-1]['Close Price'],
        })

    if not trades:
        return 0, 0, 0

    trades_df = pd.DataFrame(trades)
    trades_df['profit_pct'] = ((trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price']) * 100

    total_return_pct = trades_df['profit_pct'].sum()
    win_rate = (trades_df['profit_pct'] > 0).mean() * 100 if not trades_df.empty else 0

    # --- Max Drawdown Calculation ---
    returns = df['Close Price'].pct_change().fillna(0)
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (running_max - cumulative_returns) / running_max
    max_drawdown = drawdown.max() * 100

    return total_return_pct, win_rate, max_drawdown

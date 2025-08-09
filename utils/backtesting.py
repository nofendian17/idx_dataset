import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pandas.tseries.offsets import BDay

# Constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # Default risk-free rate for Sharpe/Sortino calculation
TRANSACTION_COST_PCT = 0.0  # Default transaction cost percentage

def _validate_input_data(df: pd.DataFrame) -> bool:
    """Validates that the input DataFrame has the required columns."""
    return {'buy_signal', 'sell_signal', 'Close Price'}.issubset(df.columns)

def _calculate_trades(df: pd.DataFrame) -> List[Dict]:
    """Calculates trade entries and exits from strategy signals."""
    df['position'] = np.where(df['buy_signal'].notna(), 1, np.nan)
    df['position'] = np.where(df['sell_signal'].notna(), 0, df['position'])
    df['position'] = df['position'].ffill()
    df['position'] = df['position'].fillna(0)

    df['trade'] = df['position'].diff()

    trades = []
    entry_price = None
    entry_date = None

    for i, row in df.iterrows():
        if row['trade'] == 1:  # Entry
            entry_price = row['buy_signal']
            entry_date = row['Date']
        elif row['trade'] == -1 and entry_price is not None:  # Exit
            exit_price = row['sell_signal']
            exit_date = row['Date']
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
            })
            entry_price = None  # Reset for next trade

    # If the last position is still open, close it at the last price
    if entry_price is not None:
        trades.append({
            'entry_date': entry_date,
            'exit_date': df.iloc[-1]['Date'],
            'entry_price': entry_price,
            'exit_price': df.iloc[-1]['Close Price'],
        })

    return trades

def _calculate_trade_metrics(trades: List[Dict]) -> Dict:
    """Calculates metrics based on trade history."""
    if not trades:
        return {
            'total_return_pct': 0,
            'win_rate_pct': 0,
            'num_trades': 0,
            'avg_trade_pct': 0,
            'profit_factor': 0,
        }

    trades_df = pd.DataFrame(trades)
    trades_df['profit_pct'] = ((trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price']) * 100

    total_return_pct = trades_df['profit_pct'].sum()
    win_rate = (trades_df['profit_pct'] > 0).mean() * 100 if not trades_df.empty else 0
    num_trades = len(trades_df)
    avg_trade_pct = trades_df['profit_pct'].mean()

    # Calculate profit factor
    winning_trades = trades_df[trades_df['profit_pct'] > 0]
    losing_trades = trades_df[trades_df['profit_pct'] <= 0]
    gross_profit = winning_trades['profit_pct'].sum()
    gross_loss = losing_trades['profit_pct'].sum()
    profit_factor = -gross_profit / gross_loss if gross_loss != 0 else float('inf')

    return {
        'total_return_pct': total_return_pct,
        'win_rate_pct': win_rate,
        'num_trades': num_trades,
        'avg_trade_pct': avg_trade_pct,
        'profit_factor': profit_factor,
    }

def _calculate_portfolio_returns(df: pd.DataFrame, trades: List[Dict]) -> pd.Series:
    """Calculates portfolio returns based on trade history."""
    portfolio_returns = pd.Series(0.0, index=df.index, dtype=float)

    for trade in trades:
        entry_idx = df[df['Date'] == trade['entry_date']].index[0]
        exit_idx = df[df['Date'] == trade['exit_date']].index[0]
        trade_return = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
        portfolio_returns.iloc[exit_idx] = trade_return

    return portfolio_returns

def _calculate_risk_metrics(portfolio_returns: pd.Series, df: pd.DataFrame) -> Dict:
    """Calculates risk-adjusted metrics."""
    daily_returns = portfolio_returns.replace(0, np.nan).dropna()

    if len(daily_returns) > 1:
        # Annualize the metrics
        annualized_return = (1 + portfolio_returns).prod() ** (TRADING_DAYS_PER_YEAR/len(portfolio_returns)) - 1
        annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe ratio
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

        # Sortino ratio (only penalize downside volatility)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0
    else:
        sharpe_ratio = 0
        sortino_ratio = 0

    # Max Drawdown
    returns = df['Close Price'].pct_change().fillna(0)
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (running_max - cumulative_returns) / running_max
    max_drawdown = drawdown.max() * 100

    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown_pct': max_drawdown,
    }

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamples the data to the specified timeframe.

    Args:
        df: DataFrame with datetime index and price data
        timeframe: Target timeframe ('daily', 'weekly', 'monthly')

    Returns:
        Resampled DataFrame
    """
    if 'Date' in df.columns:
        df = df.set_index('Date')

    if timeframe == 'daily':
        return df
    elif timeframe == 'weekly':
        return df.resample('W').last()
    elif timeframe == 'monthly':
        return df.resample('M').last()
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

def calculate_backtest_metrics(df: pd.DataFrame,
                             transaction_cost_pct: float = TRANSACTION_COST_PCT,
                             risk_free_rate: float = RISK_FREE_RATE,
                             timeframe: Optional[str] = None) -> Dict:
    """
    Calculates and returns comprehensive backtesting metrics.

    Args:
        df: DataFrame containing strategy signals and price data
        transaction_cost_pct: Percentage cost per trade (default: 0%)
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculation (default: 0%)
        timeframe: Optional timeframe for analysis ('daily', 'weekly', 'monthly')

    Returns:
        Dictionary containing all calculated metrics
    """
    # Resample data if timeframe is specified
    if timeframe:
        df = resample_data(df, timeframe)
    if not _validate_input_data(df):
        return {
            'total_return_pct': 0,
            'win_rate_pct': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'num_trades': 0,
            'avg_trade_pct': 0,
            'profit_factor': 0,
            'trades': []
        }

    trades = _calculate_trades(df)
    trade_metrics = _calculate_trade_metrics(trades)
    portfolio_returns = _calculate_portfolio_returns(df, trades)
    risk_metrics = _calculate_risk_metrics(portfolio_returns, df)

    return {
        'total_return_pct': trade_metrics['total_return_pct'],
        'win_rate_pct': trade_metrics['win_rate_pct'],
        'max_drawdown_pct': risk_metrics['max_drawdown_pct'],
        'sharpe_ratio': risk_metrics['sharpe_ratio'],
        'sortino_ratio': risk_metrics['sortino_ratio'],
        'num_trades': trade_metrics['num_trades'],
        'avg_trade_pct': trade_metrics['avg_trade_pct'],
        'profit_factor': trade_metrics['profit_factor'],
        'trades': trades
    }

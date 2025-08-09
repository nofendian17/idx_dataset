import unittest
import pandas as pd
import numpy as np
from stock_analyzer import (
    load_and_prepare_data,
    calculate_indicators,
    determine_trend_and_strength,
    identify_market_phase,
    generate_signals
)
from unittest.mock import patch, mock_open

class TestStockAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.sample_data = {
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'Stock Code': ['BBNI', 'BBNI', 'BBNI', 'BBNI'],
            'Open Price': [100, 110, 120, 130],
            'High Price': [110, 120, 130, 140],
            'Low Price': [90, 100, 110, 120],
            'Close Price': [105, 115, 125, 135],
            'Volume': [1000, 1100, 1200, 1300]
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_load_and_prepare_data(self):
        """Test the data loading and preparation function."""
        mock_csv_data = (
            "Date,Stock Code,Board,Previous Price,Last Price,Open Price,High Price,Low Price,Volume,Value\n"
            "2023-01-01,BBNI,RG,100,105,101,106,100,1000,105000"
        )

        with patch('os.listdir', return_value=['stock_data_2023-01-01.csv']):
            with patch('builtins.open', mock_open(read_data=mock_csv_data)):
                df = load_and_prepare_data('dummy_dir')
                self.assertIsNotNone(df)
                self.assertIn('Close Price', df.columns)
                self.assertEqual(df.iloc[0]['Stock Code'], 'BBNI')

    def test_calculate_indicators(self):
        """Test that indicators are calculated and added as columns."""
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        data = {
            'Date': dates, 'Stock Code': ['BBNI'] * 60,
            'Open Price': np.random.uniform(100, 200, 60),
            'High Price': np.random.uniform(150, 250, 60),
            'Low Price': np.random.uniform(80, 180, 60),
            'Close Price': np.random.uniform(120, 220, 60),
            'Volume': np.random.uniform(1000, 5000, 60)
        }
        df_large = pd.DataFrame(data)

        df_indicators = calculate_indicators(df_large)
        self.assertIn('SMA50', df_indicators.columns)
        self.assertIn('MACD_line', df_indicators.columns)
        self.assertIn('ADX', df_indicators.columns)
        self.assertFalse(df_indicators['SMA50'].isnull().all())

    def test_determine_trend_and_strength(self):
        """Test the trend and strength determination logic."""
        df = self.df.copy()
        df['SMA50'] = [100, 110, 120, 130]
        df['SMA200'] = [90, 100, 110, 120]
        df['SMA50_slope'] = [np.nan, 10, 10, 10]
        df['ADX'] = [15, 22, 28, 30]

        df_trend = determine_trend_and_strength(df)

        self.assertEqual(df_trend.iloc[2]['Trend'], 'Uptrend')
        self.assertEqual(df_trend.iloc[1]['Strength'], 'Moderate')
        self.assertEqual(df_trend.iloc[2]['Strength'], 'Strong')

    def test_identify_market_phase(self):
        """Test the market phase identification logic."""
        df = self.df.copy()
        df['Trend'] = ['Sideways', 'Downtrend', 'Uptrend', 'Uptrend']
        df['Strength'] = ['Weak', 'Strong', 'Strong', 'Strong']
        df['MACD_line'] = [0, -1, 1, 2]
        df['Signal_line'] = [0, 0, 0, 1]

        df_phase = identify_market_phase(df)

        self.assertEqual(df_phase.iloc[1]['Phase'], 'Declining')
        self.assertEqual(df_phase.iloc[2]['Phase'], 'Advancing')

    def test_generate_signals(self):
        """Test the signal generation logic."""
        df = self.df.copy()
        df['Trend'] = ['Sideways', 'Downtrend', 'Uptrend', 'Uptrend']
        df['Strength'] = ['Weak', 'Strong', 'Moderate', 'Strong']
        df['Phase'] = ['Sideways', 'Declining', 'Accumulation', 'Advancing']
        df['MACD_line'] = [0, -5, 5, 10]
        df['Signal_line'] = [1, -4, 4, 8]

        df_signal = generate_signals(df)

        self.assertEqual(df_signal.iloc[0]['Signal'], 'Neutral')
        self.assertEqual(df_signal.iloc[1]['Signal'], 'Sell')
        self.assertEqual(df_signal.iloc[2]['Signal'], 'Buy')
        self.assertEqual(df_signal.iloc[3]['Signal'], 'Buy')

if __name__ == '__main__':
    unittest.main()

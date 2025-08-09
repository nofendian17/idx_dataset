import unittest
from unittest.mock import patch, Mock
import os
import pandas as pd
import requests
from datetime import datetime
from scraper import fetch_stock_data_with_retries, save_to_csv, get_default_date

class TestScraper(unittest.TestCase):

    @patch('scraper.requests.get')
    def test_fetch_stock_data_success(self, mock_get):
        """Test successful data fetching on the first attempt."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': [{'list_stock_code': 'BBNI'}]}
        mock_get.return_value = mock_response

        data = fetch_stock_data_with_retries('2023-01-01')
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['list_stock_code'], 'BBNI')
        mock_get.assert_called_once()

    @patch('scraper.requests.get')
    def test_fetch_stock_data_with_retries(self, mock_get):
        """Test that the scraper retries on request failure and succeeds."""
        mock_failure = requests.exceptions.Timeout("Timed out")

        mock_success = Mock()
        mock_success.status_code = 200
        mock_success.json.return_value = {'data': [{'list_stock_code': 'BBRI'}]}

        # The side_effect will raise an exception on the first call, and return a mock object on the second
        mock_get.side_effect = [mock_failure, mock_success]

        with patch('scraper.time.sleep', return_value=None):
            data = fetch_stock_data_with_retries('2023-01-02')

        self.assertIsNotNone(data)
        self.assertEqual(data[0]['list_stock_code'], 'BBRI')
        self.assertEqual(mock_get.call_count, 2)

    @patch('scraper.requests.get')
    def test_fetch_stock_data_fails_after_max_retries(self, mock_get):
        """Test that fetching returns None after all retries fail."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        with patch('scraper.time.sleep', return_value=None):
            data = fetch_stock_data_with_retries('2023-01-03')

        self.assertIsNone(data)
        self.assertEqual(mock_get.call_count, 3)

    @patch('scraper.requests.get')
    def test_fetch_stock_data_http_error_no_retry(self, mock_get):
        """Test that client errors like 404 do not trigger retries."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        data = fetch_stock_data_with_retries('2023-01-04')

        self.assertIsNone(data)
        mock_get.assert_called_once()

    def test_save_to_csv(self):
        """Test saving data to a CSV file."""
        test_data = [{
            'date': '2023-01-05', 'list_stock_code': 'TEST', 'board': 'RG',
            'previous': 1000, 'last': 1050, 'open': 1010, 'high': 1060, 'low': 1000,
            'tot_volume': 10000, 'tot_value': 10500000
        }]
        date = '2023-01-05'
        data_dir = 'test_data'

        with patch('scraper.DATA_DIR', data_dir):
            os.makedirs(data_dir, exist_ok=True)
            filename = save_to_csv(date, test_data)

            self.assertTrue(os.path.exists(filename))

            df = pd.read_csv(filename)
            self.assertEqual(df.iloc[0]['Stock Code'], 'TEST')
            self.assertEqual(df.iloc[0]['Last Price'], 1050)

            os.remove(filename)
            os.rmdir(data_dir)

    def test_get_default_date(self):
        """Test that the default date is in the correct format."""
        default_date = get_default_date()
        self.assertIsNotNone(datetime.strptime(default_date, '%Y-%m-%d'))

if __name__ == '__main__':
    unittest.main()

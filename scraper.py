import requests
import csv
import argparse
import os
import sys
import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone

# --- Constants ---
JAKARTA_TZ = timezone(timedelta(hours=7))
API_BASE_URL = "https://imq21.com/daftar-saham"
DEFAULT_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DATA_DIR = "data"

# CSV Field Names
CSV_FIELDNAMES = [
    'Date', 'Stock Code', 'Board', 'Previous Price', 'Last Price',
    'Open Price', 'High Price', 'Low Price', 'Volume', 'Value'
]

# API Field Mapping
API_FIELD_MAPPING = {
    'date': 'Date',
    'list_stock_code': 'Stock Code',
    'board': 'Board',
    'previous': 'Previous Price',
    'last': 'Last Price',
    'open': 'Open Price',
    'high': 'High Price',
    'low': 'Low Price',
    'tot_volume': 'Volume',
    'tot_value': 'Value'
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def get_default_date() -> str:
    """Return today's date in Asia/Jakarta timezone."""
    return datetime.now(JAKARTA_TZ).strftime("%Y-%m-%d")

def fetch_stock_data_with_retries(date: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch stock data from the API with a retry mechanism.

    Args:
        date: The date for which to fetch data (YYYY-MM-DD).

    Returns:
        A list of dictionaries containing stock data, or None on failure.
    """
    url = f"{API_BASE_URL}?date={date}&_={int(time.time() * 1000)}"
    headers = {
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'referer': f'{API_BASE_URL}?date={date}',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                return data['data']
            if isinstance(data, list):
                return data

            logging.warning(f"Unexpected API response format for date {date}: {data}")
            return None

        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error for date {date}: {e.response.status_code} {e.response.reason}")
            if e.response.status_code in [403, 404]:
                logging.info(f"Stopping retries for date {date} due to client error.")
                break  # Don't retry on client errors like Not Found
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for date {date} on attempt {attempt + 1}/{MAX_RETRIES}: {e}")

        if attempt < MAX_RETRIES - 1:
            logging.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

    logging.critical(f"Failed to fetch data for {date} after {MAX_RETRIES} attempts.")
    return None

def save_to_csv(date: str, data: List[Dict[str, Any]]) -> str:
    """
    Save stock data to a CSV file inside the data directory.

    Args:
        date: The date of the data.
        data: A list of stock data dictionaries.

    Returns:
        The path to the saved CSV file.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"stock_data_{date}.csv")

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()

            for stock in data:
                row_data = {}
                for api_field, csv_field in API_FIELD_MAPPING.items():
                    row_data[csv_field] = stock.get(api_field, 0 if csv_field in ['Previous Price', 'Last Price', 'Open Price', 'High Price', 'Low Price', 'Volume', 'Value'] else 'N/A')

                # Special handling for date
                if 'Date' in row_data:
                    row_data['Date'] = stock.get('date', date)

                writer.writerow(row_data)

        logging.info(f"Successfully saved {len(data)} records to {filename}")
    except IOError as e:
        logging.error(f"Could not write to file {filename}: {e}")
        raise

    return filename

def main():
    """Main function to run the scraper from the command line."""
    parser = argparse.ArgumentParser(
        description='Fetch IDX stock market data for a given date.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'date',
        type=str,
        nargs='?',
        default=get_default_date(),
        help='Trading date (format: YYYY-MM-DD)'
    )
    args = parser.parse_args()

    logging.info(f"Starting stock data fetch for date: {args.date}")

    stock_data = fetch_stock_data_with_retries(args.date)

    if stock_data is None:
        logging.error("Failed to fetch stock data. API request ultimately failed.")
        sys.exit(1)
    
    if not stock_data:
        logging.info(f"No stock data available for {args.date}. It might be a weekend or holiday.")
        sys.exit(0)

    try:
        save_to_csv(args.date, stock_data)
    except IOError:
        sys.exit(1)  # Exit if file saving fails

if __name__ == "__main__":
    main()

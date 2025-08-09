import argparse
import logging
import sys
from datetime import datetime, date, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the necessary functions from the refactored scraper
from scraper import fetch_stock_data_with_retries, save_to_csv, get_default_date

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def daterange(start_date: date, end_date: date):
    """Generate a sequence of dates."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def fetch_and_save(trade_date: date) -> Optional[str]:
    """
    Fetches and saves stock data for a single day.

    Args:
        trade_date: The date to fetch data for.

    Returns:
        The date string if successful, None otherwise.
    """
    date_str = trade_date.strftime('%Y-%m-%d')
    logging.info(f"Requesting data for {date_str}...")

    stock_data = fetch_stock_data_with_retries(date_str)

    if stock_data:
        try:
            save_to_csv(date_str, stock_data)
            return date_str
        except IOError:
            return None
    elif stock_data is None:
        logging.warning(f"Fetching ultimately failed for {date_str}.")
    else: # Empty list
        logging.info(f"No data found for {date_str} (weekend/holiday).")

    return None

def main():
    """Main function to run the bulk fetcher."""
    parser = argparse.ArgumentParser(
        description='Fetch historical IDX stock data in parallel.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    parser.add_argument(
        '--start_date',
        type=str,
        default=three_years_ago,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        default=get_default_date(),
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=10,
        help='Number of parallel threads to use for fetching.'
    )

    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    except ValueError:
        logging.error("Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)

    if start_date > end_date:
        logging.error("Start date cannot be after end date.")
        sys.exit(1)

    dates_to_fetch = list(daterange(start_date, end_date))
    total_dates = len(dates_to_fetch)
    logging.info(f"Starting bulk fetch for {total_dates} days from {start_date} to {end_date} with {args.max_workers} workers.")

    successful_fetches = 0
    failed_fetches = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_date = {executor.submit(fetch_and_save, dt): dt for dt in dates_to_fetch}

        for future in as_completed(future_to_date):
            result = future.result()
            if result:
                successful_fetches += 1
            else:
                failed_fetches += 1

            # Log progress
            completed_count = successful_fetches + failed_fetches
            if completed_count % 20 == 0 or completed_count == total_dates:
                logging.info(f"Progress: {completed_count}/{total_dates} days processed.")

    logging.info("Bulk fetch finished.")
    logging.info(f"Summary: {successful_fetches} successful, {failed_fetches} failed or no data.")

if __name__ == "__main__":
    main()

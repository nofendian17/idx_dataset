#!/bin/bash

# This script fetches stock data for a specified period using the efficient bulk_fetcher.py script.

# --- Configuration ---
# Set the default start date (e.g., "3 years ago", "1 year ago")
START_DATE_DEFAULT=$(date -d "3 years ago" +%Y-%m-%d)
# Set the end date to today
END_DATE_DEFAULT=$(date +%Y-%m-%d)
# Number of parallel workers
MAX_WORKERS=10

# --- Script Execution ---
echo "Starting bulk data fetch..."
echo "This process will run in the background. Check bulk_fetcher.log for progress."

# Run the Python bulk fetcher script
# Use --start_date and --end_date to override the defaults if needed
python3 bulk_fetcher.py \
    --start_date "$START_DATE_DEFAULT" \
    --end_date "$END_DATE_DEFAULT" \
    --max_workers "$MAX_WORKERS" > bulk_fetcher.log 2>&1

echo "Bulk fetch script has been executed. See bulk_fetcher.log for details."

#!/bin/bash

# This script fetches stock data for each day from one year ago until today.

# Get the start date (3 years ago)
START_DATE=$(date -d "3 years ago" +%Y-%m-%d)

# Get the end date (today)
END_DATE=$(date +%Y-%m-%d)

# Convert dates to seconds since epoch for comparison
START_SEC=$(date -d "$START_DATE" +%s)
END_SEC=$(date -d "$END_DATE" +%s)

CURRENT_SEC=$START_SEC

echo "Fetching data from $START_DATE to $END_DATE"

while [ $CURRENT_SEC -le $END_SEC ]; do
    CURRENT_DATE=$(date -d "@$CURRENT_SEC" +%Y-%m-%d)
    echo "Fetching data for: $CURRENT_DATE"
    python3 scraper.py "$CURRENT_DATE"
    CURRENT_SEC=$((CURRENT_SEC + 86400)) # Increment by one day (86400 seconds)
done

echo "Script finished."

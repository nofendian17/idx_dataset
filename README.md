# IDX Stock Scraper

This project contains a Python script to scrape daily stock data from the Indonesia Stock Exchange (IDX) via the `imq21.com` website. The data is saved into CSV files.

## Features

- Fetches all listed stocks for a given date.
- Saves data in a structured CSV format inside the `data/` directory.
- Defaults to the current date in Asia/Jakarta timezone if no date is specified.
- Includes a GitHub Actions workflow to automatically fetch and commit the data daily.

## Usage

### Prerequisites

- Python 3
- `requests` library

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nofendian17/idx_dataset.git
    cd idx_dataset
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Scraper

You can run the scraper with an optional date argument. If no date is provided, it will use the current date in Jakarta.

-   **To fetch data for the current day:**
    ```bash
    python scraper.py
    ```

-   **To fetch data for a specific date:**
    ```bash
    python scraper.py YYYY-MM-DD
    ```
    For example:
    ```bash
    python scraper.py 2024-01-15
    ```

The script will create a `data/` directory if it doesn't exist and save the data to a file named `stock_data_YYYY-MM-DD.csv`.

## Automation

This repository is configured with a GitHub Actions workflow (`.github/workflows/fetch_stocks.yml`) that automates the data fetching process.

-   The workflow runs from Monday to Friday at 11:00 UTC (18:00 WIB).
-   It executes the `scraper.py` script to get the latest stock data.
-   The new CSV file is then automatically committed and pushed to the repository.
-   The workflow can also be triggered manually from the Actions tab in the GitHub repository.

## Data Format

The output CSV file has the following columns:

-   `Date`: The date of the stock data.
-   `Stock Code`: The stock ticker symbol.
-   `Board`: The stock exchange board.
-   `Previous Price`: The previous day's closing price.
-   `Last Price`: The last traded price.
-   `Open Price`: The opening price.
-   `High Price`: The highest price of the day.
-   `Low Price`: The lowest price of the day.
-   `Volume`: The total volume of shares traded.
-   `Value`: The total value of transactions.

## License and Usage

The data collected by this scraper is intended for educational and research purposes only.

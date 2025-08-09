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

## Bulk Data Fetching

This project includes a `bulk_fetcher.py` script that allows you to download historical stock data for a specified date range in parallel. This is useful for populating the dataset over a longer period.

### Running the Bulk Fetcher

You can run the bulk fetcher with optional start and end dates, and specify the number of worker threads to use.

-   **To fetch data for a date range (defaults to last 3 years):**
    ```bash
    python bulk_fetcher.py --start_date YYYY-MM-DD --end_date YYYY-MM-DD --max_workers N
    ```
    For example, to fetch data for January 2024 with 10 workers:
    ```bash
    python bulk_fetcher.py --start_date 2024-01-01 --end_date 2024-01-31 --max_workers 10
    ```

## Data Analysis

The `stock_analyzer.py` script performs in-depth technical analysis on the collected stock data. It calculates various indicators, identifies trends, market phases, and generates trading signals.

### Running the Analyzer

To perform analysis on the data in the `data/` directory, simply run:
```bash
python stock_analyzer.py
```
The script will output the latest analysis results for each stock.

## Interactive Dashboard

A Streamlit application (`streamlit_app.py`) is provided to visualize the stock data and analysis results. It offers an interactive dashboard with the following pages:

-   **Daily Market Summary:** Overview of market performance.
-   **Stock Trend Analysis:** Detailed trend analysis for individual stocks.
-   **Strategy Analysis:** Evaluation of trading strategies.
-   **Price Prediction:** Potential price forecasting (if implemented).
-   **Comprehensive Trend Analysis:** In-depth trend insights.

### Running the Dashboard

To launch the dashboard, ensure you have Streamlit installed (`pip install streamlit`) and then run:
```bash
streamlit run streamlit_app.py
```
This will open the dashboard in your web browser.

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

import requests
import csv
import argparse
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone

JAKARTA_TZ = timezone(timedelta(hours=7))

def get_default_date() -> str:
    """Return today's date in Asia/Jakarta timezone."""
    return datetime.now(JAKARTA_TZ).strftime("%Y-%m-%d")

def fetch_stock_data(date: str) -> Optional[List[Dict]]:
    """Fetch stock data from API."""
    url = f"https://imq21.com/daftar-saham?date={date}&_=1754526155571"
    headers = {
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'accept-language': 'en-US,en;q=0.9,id;q=0.8',
        'dnt': '1',
        'priority': 'u=1, i',
        'referer': f'https://imq21.com/daftar-saham?date={date}',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            return data['data']
        if isinstance(data, list):
            return data
        print("Unexpected API response format.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_to_csv(date: str, data: List[Dict]) -> str:
    """Save stock data to CSV inside data/ folder."""
    filename = f"data/stock_data_{date}.csv"
    fieldnames = [
        'Date', 'Stock Code', 'Board', 
        'Previous Price', 'Last Price', 
        'Open Price', 'High Price', 'Low Price',
        'Volume', 'Value'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for stock in data:
            writer.writerow({
                'Date': stock.get('date', date),
                'Stock Code': stock.get('list_stock_code', 'N/A'),
                'Board': stock.get('board', 'N/A'),
                'Previous Price': stock.get('previous', 0),
                'Last Price': stock.get('last', 0),
                'Open Price': stock.get('open', 0),
                'High Price': stock.get('high', 0),
                'Low Price': stock.get('low', 0),
                'Volume': stock.get('tot_volume', 0),
                'Value': stock.get('tot_value', 0)
            })
    
    return filename

def main():
    parser = argparse.ArgumentParser(description='Fetch stock market data for a given date.')
    parser.add_argument('date', type=str, nargs='?', default=get_default_date(),
                        help='Trading date (format: YYYY-MM-DD), defaults to today in Asia/Jakarta')
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    stock_data = fetch_stock_data(args.date)
    if stock_data:
        filename = save_to_csv(args.date, stock_data)
        print(f"Successfully saved {len(stock_data)} records to {filename}")
    else:
        print("Failed to fetch or save stock data")
        sys.exit(1)

if __name__ == "__main__":
    main()

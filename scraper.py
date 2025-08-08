import requests
import csv
import argparse
from typing import List, Dict, Optional, Union

def fetch_stock_data(date: str) -> Optional[List[Union[Dict, list]]]:
    url = f"https://imq21.com/daftar-saham?date={date}&_=1754526155571"
    headers = {
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'accept-language': 'en-US,en;q=0.9,id;q=0.8',
        'referer': f'https://imq21.com/daftar-saham?date={date}',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_data = response.json()

        # Extract data if nested
        if isinstance(json_data, dict) and "data" in json_data:
            data = json_data["data"]
        else:
            data = json_data

        if not isinstance(data, list):
            print("Unexpected JSON structure:", json_data)
            return None

        print(f"Fetched {len(data)} rows")
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_to_csv(date: str, data: List[Union[Dict, list]]) -> str:
    filename = f"stock_data_{date}.csv"
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
            if isinstance(stock, dict):
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
            elif isinstance(stock, list):
                writer.writerow({
                    'Date': date,
                    'Stock Code': stock[0] if len(stock) > 0 else 'N/A',
                    'Board': stock[1] if len(stock) > 1 else 'N/A',
                    'Previous Price': stock[2] if len(stock) > 2 else 0,
                    'Last Price': stock[3] if len(stock) > 3 else 0,
                    'Open Price': stock[4] if len(stock) > 4 else 0,
                    'High Price': stock[5] if len(stock) > 5 else 0,
                    'Low Price': stock[6] if len(stock) > 6 else 0,
                    'Volume': stock[7] if len(stock) > 7 else 0,
                    'Value': stock[8] if len(stock) > 8 else 0
                })
    
    return filename

def main():
    parser = argparse.ArgumentParser(description='Fetch stock market data for a given date.')
    parser.add_argument('date', type=str, help='Trading date (format: YYYY-MM-DD)')
    args = parser.parse_args()

    if stock_data := fetch_stock_data(args.date):
        filename = save_to_csv(args.date, stock_data)
        print(f"Successfully saved {len(stock_data)} records to {filename}")
    else:
        print("No stock data found or failed to fetch.")

if __name__ == "__main__":
    main()

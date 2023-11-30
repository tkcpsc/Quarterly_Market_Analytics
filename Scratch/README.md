# Thomas Kudey

## Overview

This Python script is designed to perform two tasks:
1. Most Active Tickers Retrieval:
Scrapes the most active stock tickers from Yahoo Finance using web scraping techniques (requests and BeautifulSoup).
Defines a function (getTopTickers) to extract symbols from the most active list.
2. Historical Stock Data Retrieval:
Utilizes the Yahoo Finance API (yfinance) to fetch historical stock data for the retrieved tickers.
Retrieves closing prices for the last 5 years and stores the data in a CSV file named MarketData.csv.

## Dependencies
Make sure you have the following Python packages installed:
- requests
- BeautifulSoup
- pandas
- yfinance
- datetime

You can install them using the following commands:

- Python 2.x:
```
pip install requests && 
pip install beautifulsoup4 && 
pip install pandas && 
pip install yfinance && 
pip install numpy
```
- Python 3.x
```
pip3 install requests &&
pip3 install beautifulsoup4 &&
pip3 install pandas && 
pip3 install yfinance && 
pip3 install numpy
```

## Usage

1. Clone the repository or download the script (stock_data_scraper.py).
2. Install the required dependencies as mentioned above.
3. Run the script using the following command:

- Python 2.x:
```
python StockMarketData.py QuarterlyData.py
```

- Python 3.x
```
python3 StockMarketData.py QuarterlyData.py
```

The scripts will first fetch the most active stock tickers, then retrieve historical stock data for the last 5 years, and save the closing prices to a CSV file (MarketData.csv) in the same directory.




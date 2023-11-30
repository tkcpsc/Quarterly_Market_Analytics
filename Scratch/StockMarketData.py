# Imports
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import yfinance as yf
from datetime import datetime, timedelta



# Function to get most active symbols from Yahoo Finance
def getTopTickers(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'lxml')

  symbols = []
  for row in soup.select('tr.simpTblRow'):
    symbol = row.select_one('td:nth-of-type(1) a').text.strip()
    symbols.append(symbol)

  return symbols




# Define the list of S&P 500 tickers.
tickers = getTopTickers('https://finance.yahoo.com/most-active')

# Set the date range for the last 5 years
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=5.2*365)).strftime('%Y-%m-%d')

# Create an empty DataFrame to store the closing prices
closing_prices_df = pd.DataFrame()

# Fetch historical stock data for each company and store closing prices
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    closing_prices_df[ticker] = stock_data['Close']


    # Display the DataFrame
# closing_prices_df.head()

# Save the DataFrame to a CSV file
closing_prices_df.to_csv('MarketData.csv', index=True)
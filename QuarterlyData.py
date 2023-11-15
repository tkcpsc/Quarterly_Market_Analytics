import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

def getTopTickers(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')  # Changed 'lxml' to 'html.parser'

    symbols = []
    for row in soup.select('tr.simpTblRow'):
        symbol = row.select_one('td:nth-of-type(1) a').text.strip()
        symbols.append(symbol)

    return symbols

# Define the list of S&P 500 tickers.
tickers = getTopTickers('https://finance.yahoo.com/most-active')


def get_quarterly_report_dates(ticker_symbols):
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each ticker symbol and fetch quarterly data
    for ticker_symbol in ticker_symbols:
        # Fetch historical data for the last 5 years with quarterly frequency
        data = yf.download(ticker_symbol, start=(pd.Timestamp.now() - pd.DateOffset(years=5)), end=pd.Timestamp.now(), interval='3mo')

        # If data is available, extract dates
        if not data.empty:
            # Extract the date index
            dates = data.index

            # Add a column for the Ticker Symbol
            df = pd.concat([df, pd.DataFrame({'Date': dates, 'Ticker': ticker_symbol})])

    return df

quarterly_report_dates = get_quarterly_report_dates(tickers)

quarterly_report_dates.to_csv('QuarterlyReports.csv', index=True)

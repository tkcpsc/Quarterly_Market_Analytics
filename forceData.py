import pandas as pd
import yfinance as yf
from datetime import datetime
from pytz import timezone
from copy import deepcopy as dc


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_df = stock_data[['Close']].reset_index()
    stock_df.columns = ['Date', 'Close']
    return stock_df

def get_past_earnings_dates(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        earnings_dates = stock.earnings_dates
        current_date = datetime.now(timezone('America/New_York'))
        past_dates_df = pd.DataFrame(earnings_dates.index[earnings_dates.index < current_date], columns=['Earnings Date'])
        past_earnings_dates_df = pd.DataFrame()
        past_earnings_dates_df["Date"] = past_dates_df['Earnings Date']
        return past_earnings_dates_df

    except Exception as e:
        return None

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
        
    df.dropna(inplace=True)
    
    return df


ticker_symbol = "AAPL"  
start_date = "2000-01-01"
end_date = "2023-11-29"
data = get_stock_data(ticker_symbol, start_date, end_date)
lookback = 7

quarters = get_past_earnings_dates(ticker_symbol)
# print(quarters)

prices = prepare_dataframe_for_lstm(data, lookback)
# print(prices)


quarters['Date'] = quarters['Date'].dt.tz_localize(None)
prices['Date'] = prices['Date'].dt.tz_localize(None)

# print(quarters['Date'])


# print(prices[prices['Date'] == "2023-11-02"])
# print(prices[prices['Date'] == "2023-08-03"])
# print(prices[prices['Date'] == "2023-05-04"])
# print(prices[prices['Date'] == "2023-02-02"])
# print(prices[prices['Date'] == "2022-10-27"])
# print(prices[prices['Date'] == "2022-07-28"])
# print(prices[prices['Date'] == "2022-04-28"])

data = pd.DataFrame(columns=prices.columns)
one = prices[prices['Date'] == "2023-11-02"]
one.reset_index(drop = True, inplace = True)
data = pd.concat([data, one])

one = prices[prices['Date'] == "2023-08-03"]
one.reset_index(drop = True, inplace = True)
data = pd.concat([data, one])

one = prices[prices['Date'] == "2023-05-04"]
one.reset_index(drop = True, inplace = True)
data = pd.concat([data, one])

one = prices[prices['Date'] == "2023-02-02"]
one.reset_index(drop = True, inplace = True)
data = pd.concat([data, one])

one = prices[prices['Date'] == "2022-10-27"]
one.reset_index(drop = True, inplace = True)
data = pd.concat([data, one])

one = prices[prices['Date'] == "2022-07-28"]
one.reset_index(drop = True, inplace = True)
data = pd.concat([data, one])

one = prices[prices['Date'] == "2022-04-28"]
one.reset_index(drop = True, inplace = True)
data = pd.concat([data, one])

# print(data.head(10))
# Assuming 'data' is your DataFrame
data.set_index('Date', inplace=True)

# Display the resulting DataFrame
# print(data.head())


# dates = ['2023-11-02', '2023-08-03', "2023-05-04", "2023-02-02", "2022-10-27", "2022-07-28", "2022-04-28"]

# Utility 
shifted_df = data
print(shifted_df)
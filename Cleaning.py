# initial script to merge earnings dates and market values. 
# this script was retired for accuracy issues.

import pandas as pd
import numpy as np

def apply_offset(timestamp, offset):
    new_timestamp = timestamp + pd.DateOffset(days=offset)
    return new_timestamp
def getPriceForValidDateAndTicker(date, ticker, market_data_path='MarketData.csv', ascending=True, limit=10):
    market_data = pd.read_csv(market_data_path, parse_dates=['Date'])
    try:
        row = market_data.loc[(market_data['Date'] == date)]
        if not row.empty:
            return row.iloc[0][ticker]
        else:
            current_date = date
            for _ in range(limit):
                if ascending:
                    current_date = market_data.loc[market_data['Date'] > current_date, 'Date'].min()
                else:
                    current_date = market_data.loc[market_data['Date'] < current_date, 'Date'].max()

                if not pd.isnull(current_date):
                    return market_data.loc[market_data['Date'] == current_date, ticker].iloc[0]
                else:
                    return None
            return None

    except Exception as e:
        return None    
def getValidDate(date, ticker, market_data_path='MarketData.csv', ascending=True, limit=10):
    market_data = pd.read_csv(market_data_path, parse_dates=['Date'])
    try:
        row = market_data.loc[(market_data['Date'] == date)]
        if not row.empty:
            return date
        else:

            current_date = date
            for _ in range(limit):
                if ascending:
                    current_date = market_data.loc[market_data['Date'] > current_date, 'Date'].min()
                else:
                    current_date = market_data.loc[market_data['Date'] < current_date, 'Date'].max()

                if not pd.isnull(current_date):
                    return current_date
                else:
                    return None
            return None

    except Exception as e:
        # print(f"Error: {e}")
        return None   
def getPriceWithOffset(date, ticker, market_data_path='MarketData.csv', ascending=True, limit=10, offset=0):
    offset_date = date

    if offset > 0:
        offset_date = apply_offset(offset_date, offset)
        offset_date = getValidDate(offset_date, ticker, ascending=True)
    elif offset < 0:
        offset_date = apply_offset(offset_date, offset)
        offset_date = getValidDate(offset_date, ticker, ascending=False)

    price = getPriceForValidDateAndTicker(offset_date, ticker, ascending=True)
    
    return price if price is not None else np.nan
  
# Example usage:
date_example = pd.to_datetime('2018-11-01')  # Replace with your desired date
ticker_example = 'TSLA'  # Replace with your desired ticker

# column_names = ["Date", "Ticker", "5_Days_Before", "4_Days_Before", "3_Days_Before", "2_Days_Before", "1_Day_Before", "Day Of", "1_Day_After", ]
example_df = [date_example.strftime('%Y-%m-%d'), 
              ticker_example,
              round(getPriceWithOffset(date_example, ticker_example, ascending=True, offset=-5),2),
              round(getPriceWithOffset(date_example, ticker_example, ascending=True, offset=-4),2),
              round(getPriceWithOffset(date_example, ticker_example, ascending=True, offset=-3),2),
              round(getPriceWithOffset(date_example, ticker_example, ascending=True, offset=-2),2),
              round(getPriceWithOffset(date_example, ticker_example, ascending=True, offset=-1),2),
              round(getPriceWithOffset(date_example, ticker_example, ascending=True, offset=-0),2),
              round(getPriceWithOffset(date_example, ticker_example, ascending=True, offset=-1),2)
]

# print(column_names)
print(example_df)

reports = pd.read_csv("QuarterlyReports.csv")

stockData = pd.DataFrame(columns=["Date", "Ticker", "5_Days_Before", "4_Days_Before", "3_Days_Before", "2_Days_Before", "1_Day_Before", "Day_Of", "1_Day_After"])

for index, row in reports.iterrows():
    date_use = pd.to_datetime(row['Date'])
    ticker_use = row['Ticker']

    new_row = pd.Series({"Date": date_use,
                         "Ticker": ticker_use,
                         "5_Days_Before": round(getPriceWithOffset(date_use, ticker_use, ascending=True, offset=-5), 2),
                         "4_Days_Before": round(getPriceWithOffset(date_use, ticker_use, ascending=True, offset=-4), 2),
                         "3_Days_Before": round(getPriceWithOffset(date_use, ticker_use, ascending=True, offset=-3), 2),
                         "2_Days_Before": round(getPriceWithOffset(date_use, ticker_use, ascending=True, offset=-2), 2),
                         "1_Day_Before": round(getPriceWithOffset(date_use, ticker_use, ascending=True, offset=-1), 2),
                         "Day_Of": round(getPriceWithOffset(date_use, ticker_use, ascending=True, offset=0), 2),
                         "1_Day_After": round(getPriceWithOffset(date_use, ticker_use, ascending=True, offset=1), 2)
                         })
    stockData.add(new_row)

stockData.fillna(0,inplace = True)

print(stockData.head(5))



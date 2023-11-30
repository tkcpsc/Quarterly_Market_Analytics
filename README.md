# Stock Price Prediction with LSTM (LIM.py)
### Overview
This Python script implements a Long Short-Term Memory (LSTM) neural network for predicting stock prices after a quarterly report using historical data. The script uses the PyTorch deep learning library for the model and Yahoo Finance (yfinance; API wrapper) for obtaining historical stock data.

### Dependencies
You can install the needed dependencies using the following shell command (modify for python3, pip3...):
```
pip install pandas numpy matplotlib torch yfinance
pip3 install pandas numpy matplotlib torch yfinance
```

### Usage
Open the script in a Python environment (e.g., Jupyter Notebook, VSCode, etc.).
Customize the script parameters such as ticker_symbol, start_date, and end_date to specify the stock and date range of interest.
```python LIM.py```
```python3 LIM.py```

### Description
  1. The script begins by fetching historical stock data using Yahoo Finance (yfinance) and extracts the closing prices.
  2. Earnings dates are obtained for the specified stock using the get_past_earnings_dates function.
  3. The data is prepared for LSTM training using the prepare_dataframe_for_lstm function, which creates a time series dataset with a specified lookback window which specifies the amount of previous days to model with.
  4. The LSTM model is defined using the LSTM class in PyTorch, and training is performed using the train_one_epoch and validate_one_epoch functions.
  5. The script then preprocesses the data, normalizes it using Min-Max scaling, and splits it into training and testing sets.
  6. The LSTM model is trained and validated for a specified number of epochs.The script generates predictions for both the training and testing sets, inversely transforms the normalized predictions to the original scale, and visualizes the results using matplotlib.

### Blockers
Due to the legacy maintainance of the earnings calendar dependency, it does not pull every record for every signle date of a quarterly report which then causes and issue of not having enough data to properly train the model. This leads to a bit of an underfit model. In the future, a thorough investigation into the earnings dates would be able to return more data for the model to train and test on.

### Citations
LSTM Model - [Gregory Hogg](https://github.com/gahogg)
Debugging - Open-AI, ChatGPT 3.5




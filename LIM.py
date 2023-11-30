# main script
# this script struggles pulling data due to the poor maintainance of the earnings reports dependency.
# README.md for more into

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yfinance as yf
from datetime import datetime
from pytz import timezone
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

ticker_symbol = "AAPL"  
start_date = "2000-01-01"
end_date = "2023-11-29"
data = get_stock_data(ticker_symbol, start_date, end_date)
lookback = 7

quarters = get_past_earnings_dates(ticker_symbol)
prices = prepare_dataframe_for_lstm(data, lookback)
# print(quarters)
# print(prices)

quarters['Date'] = quarters['Date'].dt.tz_localize(None)
prices['Date'] = prices['Date'].dt.tz_localize(None)

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

data.set_index('Date', inplace=True)
shifted_df = data
shifted_df_as_np = shifted_df.to_numpy()
# print(shifted_df_as_np)


scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
# print(shifted_df_as_np)

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]
X = dc(np.flip(X, axis=1))
split_index = int(len(X) * 0.80)
# print(split_index)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
# print(train_dataset)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    # print(x_batch.shape, y_batch.shape)
    break

model = LSTM(1, 4, 1)
model.to(device)
# print(model)

learning_rate = 0.001
num_epochs = 100
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()


with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dc(dummies[:, 0])
# print(train_predictions)

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dc(dummies[:, 0])
# print(new_y_train)

plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])
# print(test_predictions)

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])
# print(new_y_test)

plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
# early attempt to use tensorflow for the model





'''import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

df = pd.read_csv("output_stock_data.csv") 

features_columns = ["5_Days_Before", "4_Days_Before", "3_Days_Before", "2_Days_Before", "1_Day_Before", "Day_Of"]
input_columns = ["Date", "Ticker"] + features_columns
target_column = "1_Day_After"
selected_ticker = 'AAPL'

df_selected = df[df["Ticker"] == selected_ticker].copy()
df_selected.loc[:, "Date"] = pd.to_datetime(df_selected["Date"])
df_selected.loc[:, features_columns] = StandardScaler().fit_transform(df_selected[features_columns])
df_selected = df_selected.apply(pd.to_numeric, errors='coerce')
X_selected = df_selected[features_columns].values  
y_selected = df_selected[target_column].values
X_selected = np.reshape(X_selected, (X_selected.shape[0], 1, X_selected.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(64, input_shape=(1, len(features_columns)), return_sequences=True))
model.add(Dense(1))  
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")

predicted_output_selected = model.predict(X_test)
mse = mean_squared_error(y_test, predicted_output_selected.squeeze())
print(f"Mean Squared Error: {mse}")
'''
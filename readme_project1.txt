import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HHS_Unaccompanied_Alien_Children_Program.csv')

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Sort data
df = df.sort_values('Date')

# Set index
df.set_index('Date', inplace=True)

# Target column
target = 'Children in HHS Care'

# Convert target column to numeric, handling commas
df[target] = df[target].astype(str).str.replace(',', '', regex=False)
df[target] = pd.to_numeric(df[target], errors='coerce')

# Feature engineering
df['lag_1'] = df[target].shift(1)
df['lag_7'] = df[target].shift(7)

df['rolling_mean_7'] = df[target].rolling(7).mean()

df.dropna(inplace=True)

# Features
features = [
    'lag_1',
    'lag_7',
    'rolling_mean_7'
]

# Train test split
train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

mape = mean_absolute_percentage_error(y_test, predictions)

accuracy = 100 - (mape * 100)

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
print("Forecast Accuracy:", accuracy)

# Plot
plt.figure(figsize=(14,6))

plt.plot(y_test.index, y_test, label='Actual')

plt.plot(y_test.index, predictions, label='Predicted')

plt.title('Actual vs Predicted HHS Care Load')

plt.legend()

plt.show()
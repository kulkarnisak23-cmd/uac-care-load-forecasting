import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="UAC Forecasting Dashboard", layout="wide")

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.markdown("HHS UAC Program Forecasting Dashboard")

df = pd.read_csv("data/HHS_Unaccompanied_Alien_Children_Program.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].interpolate()

target = 'Children in HHS Care'

st.subheader("Dataset Preview")
st.dataframe(df.head())

fig = px.line(df, y=target, title="Children in HHS Care Trend")
st.plotly_chart(fig, use_container_width=True)

# Feature engineering
df['lag_1'] = df[target].shift(1)
df['lag_7'] = df[target].shift(7)
df['rolling_mean_7'] = df[target].rolling(7).mean()
df['rolling_std_7'] = df[target].rolling(7).std()

if 'Children transferred out of CBP custody' in df.columns and \
   'Children discharged from HHS Care' in df.columns:
    df['net_pressure'] = (
        df['Children transferred out of CBP custody'] -
        df['Children discharged from HHS Care']
    )
else:
    df['net_pressure'] = 0

df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

df.dropna(inplace=True)

features = [
    'lag_1',
    'lag_7',
    'rolling_mean_7',
    'rolling_std_7',
    'net_pressure',
    'day_of_week',
    'month'
]

train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

model_name = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "Gradient Boosting", "ARIMA"]
)

if model_name == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

elif model_name == "Gradient Boosting":
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

else:
    model = ARIMA(train[target], order=(5,1,2))
    fitted = model.fit()
    predictions = fitted.forecast(steps=len(test))

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = mean_absolute_percentage_error(y_test, predictions)
accuracy = 100 - (mape * 100)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", round(mae, 2))
c2.metric("RMSE", round(rmse, 2))
c3.metric("Forecast Accuracy %", round(accuracy, 2))

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=comparison.index, y=comparison["Actual"], mode="lines", name="Actual"))
fig2.add_trace(go.Scatter(x=comparison.index, y=comparison["Predicted"], mode="lines", name="Predicted"))

st.subheader("Actual vs Predicted")
st.plotly_chart(fig2, use_container_width=True)

st.success("Project successfully loaded.")

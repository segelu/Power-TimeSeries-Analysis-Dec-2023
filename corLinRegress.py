import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

df = pd.read_csv('PM1-6Dec.csv', index_col=0, parse_dates=True) # set the
df = df.drop(columns=['_measurement', 'machine']) # drop unused columns
dfc = df[df['_field']=='current']
dfc = dfc.rename(columns={"_value": "current"}) # rename column
dfc = dfc.drop(columns=['_field']) # drop unused columns
dfc.index.names = ['time'] # rename index
dfp = df[df['_field']=='power']
dfp = dfp.drop(columns=['_field']) # drop unused columns
dfp = dfp.rename(columns={"_value": "power"}) # rename column
dfp.index.names = ['time'] # rename index
dfcp = pd.merge(dfc, dfp, on='time')

# Add columns with year, month, and weekday name
dfcp['Year'] = dfcp.index.year
dfcp['Month'] = dfcp.index.month
dfcp['Day'] = dfcp.index.day
dfcp['Weekday Name'] = dfcp.index.day_name()
dfcp['Hour'] = dfcp.index.hour
dfcp['Minute'] = dfcp.index.minute
dfcp['Second'] = dfcp.index.second
# Display a random sampling of 5 rows
#opsd_daily.sample(5, random_state=0)
print(dfcp.head())

print(dfcp.tail())

print(dfcp.info())
print(dfcp.describe())
fig, axes = plt.subplots(nrows=1, ncols=2)
plot = dfcp['current'].plot(ax=axes[0], title="Current")
plot = dfcp['power'].plot(ax=axes[1], title="Power")
plt.subplot(1, 2, 1)
b=15
plt.hist(dfcp["power"], bins=b)
plt.xticks(ticks=np.arange(start=min(dfcp["power"]),
 step=np.ptp(dfcp["power"])/b,
 stop=np.max(dfcp["power"])))
plt.xticks(rotation=90)
plt.subplot(1, 2, 2)
b=15
plt.hist(dfcp["current"], bins=b)
plt.xticks(ticks=np.arange(start=min(dfcp["current"]),
step=np.ptp(dfcp["current"]),
stop=np.max(dfcp["current"])))
plt.xticks(rotation=90)

myFmt = mdates.DateFormatter('%H:%M:%S') # here you can format your datet
plt.gca().xaxis.set_major_formatter(myFmt)
dfcp.loc['2023-12-02 14:00:00':'2023-12-02 14:59:59', 'power'].plot()

fig,axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['power', 'current'], axes):
 sns.boxplot(data=dfcp, x='Weekday Name', y=name, ax=ax)
 ax.set_ylabel('GWh')
 ax.set_title(name)
 # Remove the automatic x-axis label from all but the bottom subplot
 if ax != axes[-1]:
  ax.set_xlabel('')

fig,axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['power', 'current'], axes):
 sns.boxplot(data=dfcp, x='Hour', y=name, ax=ax)
 ax.set_ylabel('GWh')
 ax.set_title(name)
 # Remove the automatic x-axis label from all but the bottom subplot
 if ax != axes[-1]:
  ax.set_xlabel('')

# Specify the data columns we want to include
data_columns = ['power', 'current']
print("Original series", dfcp.shape)
# Resample the data to a minute mean time series.
dfcp_minute_mean = dfcp[data_columns].resample('1min').mean()
print("Minute series", dfcp_minute_mean.shape)
# Resample the data to a hourly mean time series.
dfcp_hourly_mean = dfcp[data_columns].resample('1h').mean()
print("Hourly series", dfcp_hourly_mean.shape)
print("Hourly Mean Data:", dfcp_hourly_mean)
# Resample the data to a hourly mean time series.
dfcp_daily_mean = dfcp[data_columns].resample('1D').mean()
print("Daily series", dfcp_daily_mean.shape)

Xhm = dfcp_hourly_mean
yhm = dfcp_hourly_mean['power']
# Splitting datasets for power
X_train, X_test, y_train, y_test = train_test_split(Xhm, yhm, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Hourly Mean- MSE: {mse}, R2: {r2}")



# Start and end of the date range to extract
# Plot daily and hourly resampled time series together
start, end = '2023-12-02 00:00:00+00:00', '2023-12-03 00:00:00+00:00'
# Plot daily and hourly resampled time series together
fig,ax = plt.subplots()
ax.plot(dfcp.loc[start:end, 'power'], linewidth=0.5, label='Original (seconds)', color='cyan')
ax.plot(dfcp_minute_mean.loc[start:end, 'power'], linestyle='-', color='blue', label='Minute Mean Resample')
ax.plot(dfcp_hourly_mean.loc[start:end, 'power'],marker='o', markersize=6, linestyle='-', color='orange', label='Hourly Mean Resample')
ax.plot(dfcp_daily_mean.loc[start:end, 'power'],marker='x', markersize=8, linestyle='--', color='red', label='Daily Mean Resample')
ax.set_ylabel('Power')
ax.legend()
myFmt = mdates.DateFormatter('%D %H:%M:%S') # here you can format your da
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(rotation=90)

decomposition = seasonal_decompose(dfcp['power'], model='additive', period=(60*60*24)) # Adjust period as needed

# Plot decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(dfcp.index, decomposition.observed, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(dfcp.index, decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(dfcp.index, decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(dfcp.index, decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

trend = decomposition.trend.dropna()

# Prepare data for linear regression
time_index = np.arange(len(trend))
X_trend = pd.DataFrame({'time': time_index})
y_trend = trend.values

# Fit linear regression model
model = LinearRegression()
model.fit(X_trend, y_trend)

# Predict values
y_trend_pred = model.predict(X_trend)

# Plot the original trend and fitted line
plt.figure(figsize=(10, 5))
plt.plot(trend.index, y_trend, label='Original Trend')
plt.plot(trend.index, y_trend_pred, label='Fitted Line', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Trend Component')
plt.legend()
plt.title('Linear Regression on Decomposed Trend Component')
plt.show()

decomposition = seasonal_decompose(dfcp_hourly_mean['power'], model='additive', period=(24))  # Adjust period as needed

# Plot decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(dfcp_hourly_mean.index, decomposition.observed, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(dfcp_hourly_mean.index, decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(dfcp_hourly_mean.index, decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(dfcp_hourly_mean.index, decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

trendHr = decomposition.trend.dropna()

# Prepare data for linear regression
timeHr_index = np.arange(len(trendHr))
X_trendHr = pd.DataFrame({'time': timeHr_index})
y_trendHr = trendHr.values

# Fit linear regression model
model = LinearRegression()
model.fit(X_trendHr, y_trendHr)

# Predict values
y_trendHr_pred = model.predict(X_trendHr)

# Plot the original trend and fitted line
plt.figure(figsize=(10, 5))
plt.plot(trendHr.index, y_trendHr, label='Original Trend')
plt.plot(trendHr.index, y_trendHr_pred, label='Fitted Line', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Trend Component')
plt.legend()
plt.title('Linear Regression on Decomposed Trend Component (Hourly)')
plt.show()

from statsmodels.tsa.arima.model import ARIMA
import itertools
dfcp_hourly_mean2 = dfcp_hourly_mean.iloc[10:70,:]
train = dfcp_hourly_mean2.iloc[:50,0]
test = dfcp_hourly_mean2.iloc[49:,0]
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('Power')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Hourly Power Data")
plt.show()

y = train
ARIMAmodel = ARIMA(y, order = (5, 2, 10))
ARIMAmodel = ARIMAmodel.fit()
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('Power')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Hourly Power Data")
y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"]
plt.plot(y_pred_out, color='Blue', label = 'ARIMA Predictions')
plt.legend()
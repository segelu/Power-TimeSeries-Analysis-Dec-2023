import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('PM1-6Dec.csv')
df = df.sort_index()
scaler = MinMaxScaler()

# Fit and transform the data
df['_value_normalized'] = scaler.fit_transform(df[['_value']])
# Check current sampling rate
time_diff = df.index.to_series().diff().median()
sampling_rate = pd.Timedelta('1 second') / time_diff  # per second period
print(f"Current sampling rate: {sampling_rate} per second")
target_freq = 'H' #Per Hour
# Resample to target frequency
df._time = pd.to_datetime(df['_time'])
#df_resampled = df.set_index('_time').resample(target_freq).mean().interpolate(method='linear')
df.set_index('_time', inplace= True)
df_resampled = df.resample(target_freq).max()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['_value_normalized'], label='Original Data')
plt.plot(df_resampled.index, df_resampled['_value_normalized'], label=f'Resampled to {target_freq}')
plt.title('Original vs Resampled Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
# Decompose the time series to analyze trends, seasonality, and residuals
decomposition = seasonal_decompose(df_resampled['_value_normalized'], model='additive', period=24)  # Adjust period as needed

# Plot decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df_resampled.index, decomposition.observed, label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(df_resampled.index, decomposition.trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(df_resampled.index, decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(df_resampled.index, decomposition.resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
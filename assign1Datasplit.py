import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('PM1-6Dec.csv')
scaler = MinMaxScaler()

# Fit and transform the data
df['_value_normalized'] = scaler.fit_transform(df[['_value']])
#extracted power data
val = df.loc[df.index[df['_field']=='power'].tolist()[0]:,:]
#print(val)

pfd = val.query('_value_normalized > 0.90')
print("High Power Supply\n" + str(pfd))
afd = val.query('_value_normalized > 0.49 and _value_normalized < 0.55')
print("Average Power Supply\n" + str(afd))
lfd = val.query('_value_normalized < 0.001458')
print("Low Power Supply\n" + str(lfd))
min = val['_value'].min()
max = val['_value'].max()
mymin = df.loc[df.index[df['_value']==min]]
mymax = df.loc[df.index[df['_value']==max]]
print("Minimum Value: \n" + str(mymin) + " \nMaximum value: \n" + str(mymax))

#Confirm number of values in time series
num_values = len(df)
print(f"Number of values in time series: {num_values}")
#Identify dates over which data was captured
start_date = df['_time'].min()
end_date = df['_time'].max()
print(f"Data was captured from {start_date} to {end_date}")
#Detect periods where data was not collected
expected_dates = pd.date_range(start=start_date, end=end_date, freq='S')
missing_dates = expected_dates[~expected_dates.isin(df['_time'])]
if len(missing_dates) > 0:
    print("Missing dates in the data:")
    print(missing_dates)
else:
    print("No missing dates in the data")

plt.figure(figsize=(10, 6))
plt.plot(pfd['_time'], pfd['_value_normalized'], marker='o', linestyle='-', color='b', label='Power')
plt.title('High Power from Machine Over Time')
plt.xlabel('Time')
plt.ylabel('Power')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Visualize the time series
bdown = np.array_split(val, 30)
tplot = bdown[0]
plt.figure(figsize=(10, 6))
plt.plot(tplot['_time'], tplot['_value_normalized'], marker='o', linestyle='-', color='b', label='Power')
plt.title('Power from Machine Over Time')
plt.xlabel('Time')
plt.ylabel('Power')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
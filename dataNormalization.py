import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sample dataset
df = pd.read_csv('PM1-6Dec.csv')

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
df['_value_normalized'] = scaler.fit_transform(df[['_value']])

print("Original Data:")
print(df)

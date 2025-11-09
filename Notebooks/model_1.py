import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color_pal = sns.color_palette
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_pacf
import kagglehub
import os
import matplotlib.dates as mdates
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Download latest version
path = kagglehub.dataset_download("ddosad/dummy-truck-sales-for-time-series")

print("Path to dataset files:", path)

# Display the first few rows of data
df = pd.read_csv(path + "/Truck_sales.csv")

# convert 'Month-Year' to datetime
df['date'] = pd.to_datetime(df['Month-Year'], format='%y-%b')

# Replace month-year column with date column
df.drop(columns=['Month-Year'], inplace=True)

# list data types
print(df.dtypes)
df.head()

# initial plot of sales over time
df.plot(x = 'date', y = 'Number_Trucks_Sold',
        style='.',
        figsize=(15, 5),
        color=color_pal()[0],
        title='Number of Trucks Sold from 2003 to 2014')
plt.xlabel('Date')
plt.ylabel('Number of Trucks Sold')

# Time Series Cross Validation with plots
tss = TimeSeriesSplit(n_splits=5, test_size=12, gap=1)
df = df.sort_index()
fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

fold = 0
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train['Number_Trucks_Sold'].plot(ax=axs[fold],
                          label='Training Set',
                          title=f'Data Train/Test Split Fold {fold}')
    test['Number_Trucks_Sold'].plot(ax=axs[fold],
                         label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')

    fold += 1
plt.show()

# create derived variables using datetime features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['time_index'] = (df['date'] - df['date'].min()).dt.days
df.head()

# Create a correlation heatmap of the features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()

# Compute VIF for each feature
X = df[['year', 'month', 'month_sin', 'month_cos', 'time_index']]
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
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

# Download latest version
path = kagglehub.dataset_download("ddosad/dummy-truck-sales-for-time-series")

print("Path to dataset files:", path)

# Display the first few rows of data
df = pd.read_csv(path + "/Truck_sales.csv")
df.head()

# initial plot of sales over time
df.plot(x = 'Month-Year', y = 'Number_Trucks_Sold',
        style='.',
        figsize=(15, 5),
        color=color_pal()[0],
        title='Number of Trucks Sold from 2003 to 2014')
plt.xlabel('Month-Year')
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

# create a derived variable using the date variable
df['Year'] = df['Month-Year'].dt.year
df['Month'] = df['Month-Year'].dt.month

# Add more time-based features
df['Quarter'] = df['Month-Year'].dt.quarter
df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
df['MonthsSinceStart'] = (df['Year'] - df['Year'].min()) * 12 + df['Month']

# Import required libraries for collinearity analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Function to calculate VIF for each feature
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

# Select numerical columns for correlation analysis
numeric_cols = ['Year', 'Month', 'Quarter', 'MonthsSinceStart', 'Number_Trucks_Sold']
correlation_matrix = df[numeric_cols].corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Features')
plt.show()

# Calculate VIF
X = df[['Year', 'Month', 'Quarter', 'MonthsSinceStart']]
vif_df = calculate_vif(X)
print("\nVariance Inflation Factors:")
print(vif_df)

# Based on VIF and correlation results, let's keep only the least collinear features
# MonthsSinceStart is a good single feature that captures both Year and Month information
selected_features = ['MonthsSinceStart', 'Quarter']
print("\nSelected features after collinearity analysis:", selected_features)
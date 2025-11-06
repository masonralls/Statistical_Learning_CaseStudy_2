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
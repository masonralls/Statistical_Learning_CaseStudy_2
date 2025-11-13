import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_pacf
import kagglehub
from statsmodels.stats.outliers_influence import variance_inflation_factor


# --- Helper: robust date parsing -------------------------------------------------
def parse_month_year(series: pd.Series) -> pd.DatetimeIndex:
    """Robust parsing of common month-year formats."""
    formats = ["%d-%b-%Y", "%d-%b-%y", "%b-%Y", "%b-%y", "%m-%Y", "%m-%y", "%y-%b"]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt)
            print(f"Parsed 'Month-Year' using format: {fmt}")
            return parsed
        except Exception:
            continue
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


def run_time_series_cv(X: pd.DataFrame, y: pd.Series, n_splits=5, test_size=12, gap=1):
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    rmses = []
    last_model = None
    last_preds = None
    last_y_val = None
    for fold, (train_idx, val_idx) in enumerate(tss.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(dtype=np.float32))

        y_pred = model.predict(X_val.to_numpy(dtype=np.float32))
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f'Fold {fold} RMSE: {rmse:.2f}')
        rmses.append(rmse)

        last_model = model
        last_preds = y_pred
        last_y_val = y_val

    print(f'Average RMSE across folds: {np.mean(rmses):.2f}')
    return last_model, last_y_val, last_preds, rmses


# --- Main pipeline --------------------------------------------------------------
if __name__ == "__main__":
    path = kagglehub.dataset_download("ddosad/dummy-truck-sales-for-time-series")
    print("Path to dataset files:", path)
    df = pd.read_csv(path + "/Truck_sales.csv")

    df['date'] = parse_month_year(df['Month-Year'])
    if df['date'].isna().any():
        print("Warning: some 'Month-Year' values could not be parsed. Inspect df['Month-Year'].")
    df = df.sort_values('date').set_index('date')
    df.drop(columns=['Month-Year'], inplace=True)

    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['time_index'] = (df.index - df.index.min()).days

    plot_pacf(df['Number_Trucks_Sold'], lags=12, method='yw')
    plt.show()

    df['lag_1'] = df['Number_Trucks_Sold'].shift(1)
    df['lag_7'] = df['Number_Trucks_Sold'].shift(7)
    df['lag_10'] = df['Number_Trucks_Sold'].shift(10)
    df['lag_12'] = df['Number_Trucks_Sold'].shift(12)

    df = df.dropna()

    features = ['month_sin', 'month_cos', 'time_index', 'lag_1', 'lag_7', 'lag_10', 'lag_12']
    target = 'Number_Trucks_Sold'
    X = df[features]
    y = df[target]

    model, y_val, y_pred, rmses = run_time_series_cv(X, y)

    feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    print(feature_importance)

    low_imp = feature_importance[feature_importance['Importance'] < 0.01]['Feature'].tolist()
    if low_imp:
        print('Dropping low-importance features:', low_imp)
        features = [f for f in features if f not in low_imp]
        X = df[features]
        model, y_val, y_pred, rmses = run_time_series_cv(X, y)

    months_to_forecast = 12
    forecast_features = ['lag_1', 'lag_12']
    if not all(c in df.columns for c in forecast_features):
        raise RuntimeError('Missing lag features required for forecasting')

    model_lags = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_lags.fit(df[forecast_features].to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))

    history = df[target].tolist()
    cur_date = df.index.max()
    future_dates, future_preds = [], []
    for _ in range(months_to_forecast):
        cur_date = cur_date + pd.offsets.MonthBegin(1)
        row_vals = [history[-1], history[-12]]
        X_next = np.asarray(row_vals, dtype=np.float32).reshape(1, -1)
        y_next = float(model_lags.predict(X_next)[0])
        history.append(y_next)
        future_dates.append(cur_date)
        future_preds.append(y_next)

    forecast_df = pd.DataFrame({target: future_preds}, index=pd.DatetimeIndex(future_dates, name=df.index.name))
    print('Next forecasts:')
    print(forecast_df.head())

    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[target], label='History', color='blue')
    plt.plot(forecast_df.index, forecast_df[target], label='Forecast', color='orange')
    plt.title('Iterative Monthly Forecast (lag_1, lag_12)')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.legend()
    plt.show()

feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
print(feature_importance)

low_imp = feature_importance[feature_importance['Importance'] < 0.01]['Feature'].tolist()
if low_imp:
    print('Dropping low-importance features:', low_imp)
    features = [f for f in features if f not in low_imp]
    X = df[features]
    model, y_val, y_pred, rmses = run_time_series_cv(X, y)


months_to_forecast = 12
forecast_features = ['lag_1', 'lag_12']
if not all(c in df.columns for c in forecast_features):
    raise RuntimeError('Missing lag features required for forecasting')

model_lags = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model_lags.fit(df[forecast_features].to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))

history = df[target].tolist()
cur_date = df.index.max()
future_dates, future_preds = [], []
for _ in range(months_to_forecast):
    cur_date = cur_date + pd.offsets.MonthBegin(1)
    row_vals = [history[-1], history[-12]]
    X_next = np.asarray(row_vals, dtype=np.float32).reshape(1, -1)
    y_next = float(model_lags.predict(X_next)[0])
    history.append(y_next)
    future_dates.append(cur_date)
    future_preds.append(y_next)

forecast_df = pd.DataFrame({target: future_preds}, index=pd.DatetimeIndex(future_dates, name=df.index.name))
print('Next forecasts:')
print(forecast_df.head())

plt.figure(figsize=(15, 5))
plt.plot(df.index, df[target], label='History', color='blue')
plt.plot(forecast_df.index, forecast_df[target], label='Forecast', color='orange')
plt.title('Iterative Monthly Forecast (lag_1, lag_12)')
plt.xlabel('Date')
plt.ylabel(target)
plt.legend()
plt.show()
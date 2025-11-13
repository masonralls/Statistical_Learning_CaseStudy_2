import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_pacf
import kagglehub


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

    # Fallback: let pandas infer
    print("Falling back to generic datetime parsing for 'Month-Year'.")
    return pd.to_datetime(series, errors="coerce")


def run_time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 12,
    gap: int = 1,
):
    """
    Run time-series cross-validation with XGBoost and report RMSE per fold.
    Returns the last fitted model, a DataFrame of all CV predictions,
    and the list of fold RMSEs.
    """
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    rmses = []
    last_model = None
    fold_frames = []

    for fold, (train_idx, val_idx) in enumerate(tss.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            random_state=42,
        )
        model.fit(
            X_train.to_numpy(dtype=np.float32),
            y_train.to_numpy(dtype=np.float32),
        )

        y_pred = model.predict(X_val.to_numpy(dtype=np.float32))
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"Fold {fold} RMSE: {rmse:.2f}")
        rmses.append(rmse)

        # store fold results with original datetime index
        fold_df = pd.DataFrame(
            {"y_true": y_val.values, "y_pred": y_pred},
            index=y_val.index,
        )
        fold_frames.append(fold_df)

        last_model = model

    cv_results = pd.concat(fold_frames).sort_index()
    print(f"Average RMSE across folds: {np.mean(rmses):.2f}")
    return last_model, cv_results, rmses


# --- Main pipeline --------------------------------------------------------------
def main():
    # Load data
    path = kagglehub.dataset_download("ddosad/dummy-truck-sales-for-time-series")
    print("Path to dataset files:", path)
    df = pd.read_csv(path + "/Truck_sales.csv")

    # Parse and set date index
    df["date"] = parse_month_year(df["Month-Year"])
    if df["date"].isna().any():
        print("Warning: some 'Month-Year' values could not be parsed. Inspect df['Month-Year'].")

    df = df.sort_values("date").set_index("date")
    df.drop(columns=["Month-Year"], inplace=True)

    # Seasonal + trend features
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["time_index"] = (df.index - df.index.min()).days

    # PACF plot
    plot_pacf(df["Number_Trucks_Sold"], lags=12, method="yw")
    plt.show()

    # Lag features
    df["lag_1"] = df["Number_Trucks_Sold"].shift(1)
    df["lag_7"] = df["Number_Trucks_Sold"].shift(7)
    df["lag_10"] = df["Number_Trucks_Sold"].shift(10)
    df["lag_12"] = df["Number_Trucks_Sold"].shift(12)

    # Drop rows with missing lags
    df = df.dropna()

    target = "Number_Trucks_Sold"
    features = ["month_sin", "month_cos", "time_index", "lag_1", "lag_7", "lag_10", "lag_12"]
    X = df[features]
    y = df[target]

    # Initial time-series CV
    model, cv_results, rmses = run_time_series_cv(X, y)

    # Feature importance
    feature_importance = (
        pd.DataFrame(
            {"Feature": features, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False)
    )
    print("Feature importance (initial model):")
    print(feature_importance)

    # Drop low-importance features and re-run CV
    low_imp = feature_importance[feature_importance["Importance"] < 0.01]["Feature"].tolist()
    if low_imp:
        print("Dropping low-importance features:", low_imp)
        features = [f for f in features if f not in low_imp]
        X = df[features]
        model, cv_results, rmses = run_time_series_cv(X, y)
        feature_importance = (
            pd.DataFrame(
                {"Feature": features, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False)
        )
        print("Feature importance (after dropping low-importance features):")
        print(feature_importance)

    # --- Residual plot from CV runs --------------------------------------------
    cv_results["residual"] = cv_results["y_true"] - cv_results["y_pred"]

    plt.figure(figsize=(12, 4))
    plt.plot(cv_results.index, cv_results["residual"], marker="o", linestyle="-")
    plt.axhline(0, linestyle="--")
    plt.title("Time Series CV Residuals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.tight_layout()
    plt.show()

    # --- Forecasting with lag + seasonal + trend features ----------------------
    months_to_forecast = 12
    forecast_features = ["lag_1", "lag_12", "month_sin", "month_cos", "time_index"]
    if not all(c in df.columns for c in forecast_features):
        raise RuntimeError("Missing features required for forecasting")

    # Train forecast model on full history with these features
    model_lags = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        random_state=42,
    )
    model_lags.fit(
        df[forecast_features].to_numpy(dtype=np.float32),
        y.to_numpy(dtype=np.float32),
    )

    history = df[target].tolist()
    cur_date = df.index.max()
    future_dates, future_preds = [], []

    base_date = df.index.min()  # for consistent time_index

    for _ in range(months_to_forecast):
        # move forward one month
        cur_date = cur_date + pd.offsets.MonthBegin(1)

        # lag features from history (includes previous forecasts as we iterate)
        lag_1 = history[-1]
        lag_12 = history[-12]

        # seasonal + time features for the new date
        month = cur_date.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        time_index_future = (cur_date - base_date).days

        row_vals = [lag_1, lag_12, month_sin, month_cos, time_index_future]
        X_next = np.asarray(row_vals, dtype=np.float32).reshape(1, -1)
        y_next = float(model_lags.predict(X_next)[0])

        history.append(y_next)
        future_dates.append(cur_date)
        future_preds.append(y_next)

    forecast_df = pd.DataFrame(
        {target: future_preds},
        index=pd.DatetimeIndex(future_dates, name=df.index.name),
    )

    print("Next forecasts:")
    print(forecast_df.head())

    # Plot history + forecast
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[target], label="History")
    plt.plot(forecast_df.index, forecast_df[target], label="Forecast")
    plt.axvline(df.index.max(), linestyle="--", label="Forecast Start")
    plt.title("Iterative Monthly Forecast with Lags + Seasonality + Trend")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import json
import boto3
import io
import os
from datetime import datetime

S3_BUCKET       = os.environ.get("S3_BUCKET", "walmart-scraper-data")
FORECAST_PREFIX = "forecasts/"
FORECAST_DAYS   = 14

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed — using linear trend fallback")


# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages and lag features to daily data."""
    df = df.copy().sort_values("date")

    df["rolling_7d_avg"]  = df["avg_price"].rolling(7,  min_periods=1).mean()
    df["rolling_14d_avg"] = df["avg_price"].rolling(14, min_periods=1).mean()
    df["rolling_30d_avg"] = df["avg_price"].rolling(30, min_periods=1).mean()

    df["lag_1d"]  = df["avg_price"].shift(1)
    df["lag_7d"]  = df["avg_price"].shift(7)
    df["lag_14d"] = df["avg_price"].shift(14)

    df["price_change_1d"]  = df["avg_price"].diff(1)
    df["price_change_7d"]  = df["avg_price"].diff(7)
    df["pct_change_7d"]    = df["avg_price"].pct_change(7) * 100

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)

    return df


# ── Prophet model ─────────────────────────────────────────────────────────────
def train_prophet(df: pd.DataFrame) -> dict:
    """
    Train Prophet model on daily avg_price time series.
    Returns forecast dict with predictions for next FORECAST_DAYS days.
    """
    ts = df[["date", "avg_price"]].rename(
        columns={"date": "ds", "avg_price": "y"}
    ).dropna()

    if len(ts) < 10:
        return _linear_fallback(ts)

    if not PROPHET_AVAILABLE:
        return _linear_fallback(ts)

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.80
    )
    model.fit(ts)

    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result.columns = ["date", "forecast", "forecast_low", "forecast_high"]
    result["date"] = pd.to_datetime(result["date"])

    # Merge actuals
    result = result.merge(
        ts.rename(columns={"ds": "date", "y": "actual"}),
        on="date", how="left"
    )

    # Metrics on historical period
    hist = result[result["actual"].notna()].copy()
    mae  = np.mean(np.abs(hist["actual"] - hist["forecast"]))
    mape = np.mean(np.abs((hist["actual"] - hist["forecast"]) / hist["actual"])) * 100

    return {
        "model":   "prophet",
        "mae":     round(mae, 4),
        "mape":    round(mape, 2),
        "forecast": result.to_dict(orient="records")
    }


def _linear_fallback(ts: pd.DataFrame) -> dict:
    """Simple linear trend fallback when Prophet unavailable."""
    ts = ts.copy()
    ts["t"] = (ts["ds"] - ts["ds"].min()).dt.days
    m, b = np.polyfit(ts["t"], ts["y"], 1)

    last_date = ts["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS
    )
    last_t = ts["t"].max()
    future_t = last_t + np.arange(1, FORECAST_DAYS + 1)
    forecasts = m * future_t + b

    future_df = pd.DataFrame({
        "date":         future_dates,
        "forecast":     forecasts,
        "forecast_low": forecasts * 0.97,
        "forecast_high":forecasts * 1.03,
        "actual":       np.nan
    })

    hist_df = ts.rename(columns={"ds":"date","y":"actual"}).copy()
    hist_df["forecast"]      = m * hist_df["t"] + b
    hist_df["forecast_low"]  = hist_df["forecast"] * 0.97
    hist_df["forecast_high"] = hist_df["forecast"] * 1.03

    result = pd.concat([hist_df[["date","actual","forecast",
                                  "forecast_low","forecast_high"]],
                        future_df], ignore_index=True)

    hist = result[result["actual"].notna()]
    mae  = np.mean(np.abs(hist["actual"] - hist["forecast"]))
    mape = np.mean(np.abs((hist["actual"] - hist["forecast"]) / hist["actual"])) * 100

    return {
        "model":    "linear_fallback",
        "mae":      round(mae, 4),
        "mape":     round(mape, 2),
        "forecast": result.to_dict(orient="records")
    }


# ── Run all forecasts ─────────────────────────────────────────────────────────
def run_all_forecasts(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train one model per retailer × product combination.
    Returns combined forecast DataFrame.
    """
    all_results = []
    groups = daily_df.groupby(["retailer", "product"])

    for (retailer, product), group in groups:
        print(f"  Forecasting {retailer} - {product} ({len(group)} days)...")
        group = add_features(group.copy())

        result = train_prophet(group)
        forecast_df = pd.DataFrame(result["forecast"])
        forecast_df["retailer"] = retailer
        forecast_df["product"]  = product
        forecast_df["model"]    = result["model"]
        forecast_df["mae"]      = result["mae"]
        forecast_df["mape"]     = result["mape"]
        forecast_df["date"]     = pd.to_datetime(forecast_df["date"])

        all_results.append(forecast_df)
        print(f"    MAE: ${result['mae']:.3f} | MAPE: {result['mape']:.1f}%")

    return pd.concat(all_results, ignore_index=True)


# ── Save to S3 ────────────────────────────────────────────────────────────────
def save_forecast_to_s3(forecast_df: pd.DataFrame, run_date: str = None):
    """Save forecast CSV to S3 forecasts/ folder."""
    if run_date is None:
        run_date = datetime.utcnow().strftime("%Y-%m-%d")

    key = f"{FORECAST_PREFIX}{run_date}/price_forecasts.csv"
    s3  = boto3.client("s3")

    csv_buffer = io.StringIO()
    forecast_df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv"
    )
    print(f"Saved forecast to s3://{S3_BUCKET}/{key}")
    return key


# ── Lambda handler ────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    """Entry point for AWS Lambda."""
    from data_loader import load_all_data

    print("Loading data from S3...")
    daily_df = load_all_data()

    print("Running forecasts...")
    forecast_df = run_all_forecasts(daily_df)

    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    key = save_forecast_to_s3(forecast_df, run_date)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "run_date":        run_date,
            "forecast_records": len(forecast_df),
            "saved_to":        key
        })
    }


if __name__ == "__main__":
    from data_loader import load_all_data
    daily_df    = load_all_data()
    forecast_df = run_all_forecasts(daily_df)
    print(forecast_df.tail(20).to_string())
    save_forecast_to_s3(forecast_df)

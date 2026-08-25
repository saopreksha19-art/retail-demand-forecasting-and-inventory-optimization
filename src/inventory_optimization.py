import pandas as pd
import numpy as np
import os

FORECAST_FILE = "data/processed/forecast_predictions.csv"
MODEL_FILE = "data/processed/model_data.csv"
OUTPUT_FILE = "data/processed/inventory_recommendations.csv"

print("Loading XGBoost forecasts...")

forecast = pd.read_csv(
    FORECAST_FILE,
    low_memory=False
)

forecast["date"] = pd.to_datetime(forecast["date"])

print("Forecast shape:", forecast.shape)

print("Loading historical demand features...")

history = pd.read_csv(
    MODEL_FILE,
    low_memory=False
)

history["date"] = pd.to_datetime(history["date"])

# Keep the forecast/test period
history = history[
    history["date"] >= forecast["date"].min()
].copy()

# Select the variability features needed
history = history[
    [
        "date",
        "item_id",
        "store_id",
        "rolling_std_7"
    ]
]

# Merge forecast with historical variability
df = forecast.merge(
    history,
    on=["date", "item_id", "store_id"],
    how="left"
)

print("Merged shape:", df.shape)

# =========================================================
# INVENTORY PARAMETERS
# =========================================================

LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.645

# We model inventory coverage as 10 days of forecast demand.
# This is an explicit business assumption because the M5
# dataset does not contain actual inventory-on-hand.

INVENTORY_COVERAGE_DAYS = 10

# =========================================================
# FORECAST DEMAND
# =========================================================

df["forecast_daily_demand"] = (
    df["predicted_demand"]
    .clip(lower=0)
)

# =========================================================
# DEMAND VARIABILITY
# =========================================================

df["demand_std"] = (
    df["rolling_std_7"]
    .fillna(0)
    .clip(lower=0)
)

# =========================================================
# LEAD-TIME DEMAND
# =========================================================

df["lead_time_demand"] = (
    df["forecast_daily_demand"]
    * LEAD_TIME_DAYS
)

# =========================================================
# SAFETY STOCK
# =========================================================

df["safety_stock"] = (
    SERVICE_LEVEL_Z
    * df["demand_std"]
    * np.sqrt(LEAD_TIME_DAYS)
)

# =========================================================
# REORDER POINT
# =========================================================

df["reorder_point"] = (
    df["lead_time_demand"]
    + df["safety_stock"]
)

# =========================================================
# ESTIMATED INVENTORY
# =========================================================

df["estimated_inventory"] = (
    df["forecast_daily_demand"]
    * INVENTORY_COVERAGE_DAYS
)

# =========================================================
# DAYS OF INVENTORY
# =========================================================

df["days_until_stockout"] = np.where(
    df["forecast_daily_demand"] > 0,
    df["estimated_inventory"]
    / df["forecast_daily_demand"],
    np.inf
)

# =========================================================
# INVENTORY GAP
# =========================================================

df["inventory_gap"] = (
    df["reorder_point"]
    - df["estimated_inventory"]
)

# =========================================================
# RISK CLASSIFICATION
# =========================================================

def classify_risk(row):

    forecast = row["forecast_daily_demand"]
    inventory = row["estimated_inventory"]
    reorder = row["reorder_point"]

    if forecast <= 0:
        return "LOW"

    coverage = inventory / forecast

    if coverage < LEAD_TIME_DAYS:
        return "HIGH"

    elif inventory < reorder:
        return "MEDIUM"

    else:
        return "LOW"


df["inventory_risk"] = df.apply(
    classify_risk,
    axis=1
)

# =========================================================
# RECOMMENDED ORDER QUANTITY
# =========================================================

df["recommended_order_qty"] = (
    df["inventory_gap"]
    .clip(lower=0)
)

df["recommended_order_qty"] = np.ceil(
    df["recommended_order_qty"]
)

# =========================================================
# FINAL OUTPUT
# =========================================================

output_columns = [
    "date",
    "item_id",
    "store_id",
    "sales",
    "predicted_demand",
    "demand_std",
    "lead_time_demand",
    "safety_stock",
    "reorder_point",
    "estimated_inventory",
    "days_until_stockout",
    "recommended_order_qty",
    "inventory_risk"
]

result = df[output_columns].copy()

result = result[
    result["predicted_demand"].notna()
].copy()

os.makedirs(
    "data/processed",
    exist_ok=True
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================================================
# SUMMARY
# =========================================================

print("\n===== INVENTORY OPTIMIZATION COMPLETE =====")

print("Output shape:", result.shape)

print("\nRisk distribution:")
print(
    result["inventory_risk"]
    .value_counts()
)

print(
    "\nAverage predicted daily demand:",
    round(result["predicted_demand"].mean(), 2)
)

print(
    "Average safety stock:",
    round(result["safety_stock"].mean(), 2)
)

print(
    "Average reorder point:",
    round(result["reorder_point"].mean(), 2)
)

print(
    "Total recommended order quantity:",
    round(result["recommended_order_qty"].sum(), 0)
)

print("\nTop reorder recommendations:")

print(
    result
    .sort_values(
        "recommended_order_qty",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

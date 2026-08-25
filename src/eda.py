import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_FILE = Path("data/processed/sales_clean.csv")
OUTPUT_DIR = Path("data/processed/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading cleaned dataset...")

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

print("Shape:", df.shape)

# ---------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------

print("\n===== BASIC STATISTICS =====")

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nNumber of products:")
print(df["item_id"].nunique())

print("\nTotal units sold:")
print(df["sales"].sum())

print("\nAverage daily product demand:")
print(df["sales"].mean())

print("\nMedian demand:")
print(df["sales"].median())

print("\nMaximum daily demand:")
print(df["sales"].max())

# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\n===== MISSING VALUES =====")

missing = df.isnull().sum()

print(
    missing[
        missing > 0
    ].sort_values(ascending=False)
)

# ---------------------------------------------------------
# Zero-demand analysis
# ---------------------------------------------------------

zero_percentage = (
    (df["sales"] == 0).mean() * 100
)

print(
    f"\nPercentage of zero-demand observations: "
    f"{zero_percentage:.2f}%"
)

# ---------------------------------------------------------
# Daily sales trend
# ---------------------------------------------------------

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 5))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"]
)

plt.title("Daily Retail Sales Trend")
plt.xlabel("Date")
plt.ylabel("Units Sold")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "daily_sales_trend.png"
)

plt.close()

# ---------------------------------------------------------
# Monthly sales
# ---------------------------------------------------------

df["month"] = df["date"].dt.to_period("M")

monthly_sales = (
    df.groupby("month")["sales"]
    .sum()
    .reset_index()
)

monthly_sales["month"] = (
    monthly_sales["month"]
    .astype(str)
)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_sales["month"],
    monthly_sales["sales"]
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Units Sold")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_sales_trend.png"
)

plt.close()

# ---------------------------------------------------------
# Day-of-week analysis
# ---------------------------------------------------------

df["day_of_week"] = df["date"].dt.day_name()

weekday_sales = (
    df.groupby("day_of_week")["sales"]
    .mean()
    .reindex(
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )
)

print("\n===== AVERAGE DEMAND BY DAY =====")

print(weekday_sales)

plt.figure(figsize=(10, 5))

weekday_sales.plot(
    kind="bar"
)

plt.title("Average Demand by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Units Sold")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "weekday_demand.png"
)

plt.close()

# ---------------------------------------------------------
# Top products
# ---------------------------------------------------------

top_products = (
    df.groupby("item_id")["sales"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)

print("\n===== TOP 10 PRODUCTS =====")

print(top_products)

# ---------------------------------------------------------
# Product demand variability
# ---------------------------------------------------------

product_stats = (
    df.groupby("item_id")["sales"]
    .agg(
        total_sales="sum",
        avg_demand="mean",
        demand_std="std"
    )
)

product_stats["coefficient_variation"] = (
    product_stats["demand_std"]
    / product_stats["avg_demand"].replace(0, np.nan)
)

product_stats = product_stats.sort_values(
    "total_sales",
    ascending=False
)

print("\n===== PRODUCT DEMAND STATISTICS =====")

print(product_stats.head(10))

# ---------------------------------------------------------
# Save statistics
# ---------------------------------------------------------

product_stats.to_csv(
    OUTPUT_DIR / "product_statistics.csv"
)

daily_sales.to_csv(
    OUTPUT_DIR / "daily_sales.csv",
    index=False
)

print("\nEDA completed successfully.")

print(
    f"Results saved to: {OUTPUT_DIR}"
)

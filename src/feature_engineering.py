import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/sales_clean.csv"
OUTPUT_FILE = "data/processed/model_data.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

df["date"] = pd.to_datetime(df["date"])

# --------------------------------------------------
# 1. Select a manageable modeling subset
# --------------------------------------------------

# Use CA_1 store for the first modeling iteration
df = df[df["store_id"] == "CA_1"].copy()

# Select top 300 products by total historical sales
top_products = (
    df.groupby("item_id")["sales"]
    .sum()
    .nlargest(300)
    .index
)

df = df[df["item_id"].isin(top_products)].copy()

print("Modeling data shape:", df.shape)

# --------------------------------------------------
# 2. Calendar features
# --------------------------------------------------

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# --------------------------------------------------
# 3. Lag features
# --------------------------------------------------

group_cols = ["item_id", "store_id"]

print("Creating lag features...")

for lag in [7, 14, 28]:
    df[f"lag_{lag}"] = (
        df.groupby(group_cols)["sales"]
        .shift(lag)
    )

# --------------------------------------------------
# 4. Rolling demand features
# --------------------------------------------------

print("Creating rolling features...")

for window in [7, 14, 28]:

    df[f"rolling_mean_{window}"] = (
        df.groupby(group_cols)["sales"]
        .transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    )

    df[f"rolling_std_{window}"] = (
        df.groupby(group_cols)["sales"]
        .transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    )

# --------------------------------------------------
# 5. Demand trend
# --------------------------------------------------

df["demand_change_7"] = (
    df["lag_7"] - df["lag_14"]
)

# --------------------------------------------------
# 6. Remove rows where lag features aren't available
# --------------------------------------------------

feature_columns = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
]

df = df.dropna(subset=feature_columns)

# --------------------------------------------------
# 7. Sort data
# --------------------------------------------------

df = df.sort_values(
    ["item_id", "store_id", "date"]
).reset_index(drop=True)

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature engineering complete.")
print("Final shape:", df.shape)
print("Saved to:", OUTPUT_FILE)

print("\nFeatures:")
print(df.columns.tolist())

print("\nSample:")
print(df.head())

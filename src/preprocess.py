import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading sales data...")

sales = pd.read_csv(
    RAW_DIR / "sales_train_validation.csv"
)

calendar = pd.read_csv(
    RAW_DIR / "calendar.csv"
)

print(f"Sales shape: {sales.shape}")
print(f"Calendar shape: {calendar.shape}")

# ---------------------------------------------------------
# Identify daily sales columns
# ---------------------------------------------------------

id_columns = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

date_columns = [
    col for col in sales.columns
    if col.startswith("d_")
]

print(f"Number of daily columns: {len(date_columns)}")

# ---------------------------------------------------------
# Select a manageable subset
# ---------------------------------------------------------

# We will initially work with one store and selected products.
# This keeps development fast while preserving the real
# forecasting problem.

selected_store = "CA_1"

sales_subset = sales[
    sales["store_id"] == selected_store
].copy()

print(
    f"Rows after store selection: "
    f"{len(sales_subset):,}"
)

# ---------------------------------------------------------
# Convert wide → long
# ---------------------------------------------------------

print("Converting sales data to long format...")

sales_long = sales_subset.melt(
    id_vars=id_columns,
    value_vars=date_columns,
    var_name="d",
    value_name="sales"
)

# ---------------------------------------------------------
# Merge calendar information
# ---------------------------------------------------------

print("Merging calendar data...")

sales_long = sales_long.merge(
    calendar,
    on="d",
    how="left"
)

# ---------------------------------------------------------
# Convert date column
# ---------------------------------------------------------

sales_long["date"] = pd.to_datetime(
    sales_long["date"]
)

# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------

sales_long = sales_long.sort_values(
    ["item_id", "date"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_file = (
    PROCESSED_DIR /
    "sales_clean.csv"
)

sales_long.to_csv(
    output_file,
    index=False
)

print("\nProcessing complete.")
print(f"Final shape: {sales_long.shape}")
print(f"Saved to: {output_file}")

print("\nSample:")
print(
    sales_long[
        [
            "date",
            "item_id",
            "store_id",
            "sales"
        ]
    ].head()
)

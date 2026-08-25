import sqlite3
import pandas as pd
import os

DB_FILE = "data/processed/retail_forecasting.db"

FORECAST_FILE = "data/processed/forecast_predictions.csv"
INVENTORY_FILE = "data/processed/inventory_recommendations.csv"

os.makedirs("data/processed", exist_ok=True)

print("Creating SQLite database...")

conn = sqlite3.connect(DB_FILE)

# --------------------------------------------------
# Forecast table
# --------------------------------------------------

print("Loading forecast data...")

forecast = pd.read_csv(
    FORECAST_FILE
)

forecast["date"] = pd.to_datetime(
    forecast["date"]
)

forecast.to_sql(
    "forecast_predictions",
    conn,
    if_exists="replace",
    index=False
)

# --------------------------------------------------
# Inventory table
# --------------------------------------------------

print("Loading inventory recommendations...")

inventory = pd.read_csv(
    INVENTORY_FILE
)

inventory["date"] = pd.to_datetime(
    inventory["date"]
)

inventory.to_sql(
    "inventory_recommendations",
    conn,
    if_exists="replace",
    index=False
)

# --------------------------------------------------
# Create indexes
# --------------------------------------------------

cursor = conn.cursor()

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_forecast_item_date
ON forecast_predictions(item_id, date)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_inventory_item_date
ON inventory_recommendations(item_id, date)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_inventory_risk
ON inventory_recommendations(inventory_risk)
""")

conn.commit()

# --------------------------------------------------
# Verify
# --------------------------------------------------

print("\n===== DATABASE CREATED =====")

tables = pd.read_sql_query(
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    """,
    conn
)

print(tables)

for table in [
    "forecast_predictions",
    "inventory_recommendations"
]:

    count = pd.read_sql_query(
        f"SELECT COUNT(*) AS rows FROM {table}",
        conn
    )

    print(
        f"{table}:",
        count.iloc[0]["rows"],
        "rows"
    )

conn.close()

print("\nSaved to:")
print(DB_FILE)

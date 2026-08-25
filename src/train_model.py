import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

INPUT_FILE = "data/processed/model_data.csv"

print("Loading model data...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
df["date"] = pd.to_datetime(df["date"])

# --------------------------------------------------
# 1. TIME-BASED SPLIT
# --------------------------------------------------

train = df[df["date"] < "2016-01-01"].copy()

validation = df[
    (df["date"] >= "2016-01-01") &
    (df["date"] < "2016-04-01")
].copy()

test = df[df["date"] >= "2016-04-01"].copy()

print("\n===== DATASET SPLIT =====")
print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)

# --------------------------------------------------
# 2. FEATURES
# --------------------------------------------------

features = [
    "day_of_week",
    "day_of_month",
    "week",
    "month",
    "year",
    "is_weekend",

    "lag_7",
    "lag_14",
    "lag_28",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",

    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",

    "demand_change_7"
]

target = "sales"

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]

X_test = test[features]
y_test = test[target]

# --------------------------------------------------
# 3. NAIVE 7-DAY BASELINE
# --------------------------------------------------

print("\n===== NAIVE 7-DAY BASELINE =====")

baseline_pred = validation["lag_7"]

baseline_mae = mean_absolute_error(
    y_val,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        baseline_pred
    )
)

print(f"MAE:  {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")

# --------------------------------------------------
# 4. XGBOOST MODEL
# --------------------------------------------------

print("\n===== TRAINING XGBOOST =====")

model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# --------------------------------------------------
# 5. VALIDATION
# --------------------------------------------------

val_pred = model.predict(X_val)
val_pred = np.maximum(val_pred, 0)

val_mae = mean_absolute_error(
    y_val,
    val_pred
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_pred
    )
)

val_wape = (
    np.sum(np.abs(y_val - val_pred))
    / np.sum(np.abs(y_val))
) * 100

print("\n===== VALIDATION RESULTS =====")
print(f"MAE:  {val_mae:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"WAPE: {val_wape:.2f}%")

# --------------------------------------------------
# 6. MODEL COMPARISON
# --------------------------------------------------

improvement = (
    (baseline_mae - val_mae)
    / baseline_mae
) * 100

print("\n===== MODEL COMPARISON =====")
print(f"Naive-7 MAE : {baseline_mae:.4f}")
print(f"XGBoost MAE : {val_mae:.4f}")
print(f"Improvement  : {improvement:.2f}%")

# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance.to_string(index=False))

importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

# --------------------------------------------------
# 8. FINAL TEST EVALUATION
# --------------------------------------------------

print("\n===== TEST EVALUATION =====")

test_pred = model.predict(X_test)
test_pred = np.maximum(test_pred, 0)
prediction_output = test[
    ["date", "item_id", "store_id", "sales"]
].copy()

prediction_output["predicted_demand"] = test_pred

prediction_output.to_csv(
    "data/processed/forecast_predictions.csv",
    index=False
)

print("\nForecast predictions saved to:")
print("data/processed/forecast_predictions.csv")
test_mae = mean_absolute_error(
    y_test,
    test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)

test_wape = (
    np.sum(np.abs(y_test - test_pred))
    / np.sum(np.abs(y_test))
) * 100

print(f"Test MAE : {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test WAPE: {test_wape:.2f}%")
# =========================================================
# SAVE TRAINED MODEL
# =========================================================

import os

os.makedirs("models", exist_ok=True)

model.save_model("models/xgboost_retail_forecaster.json")

print("\nModel saved to: models/xgboost_retail_forecaster.json")

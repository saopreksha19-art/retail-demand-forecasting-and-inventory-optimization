# Retail Demand Forecasting & Inventory Optimization

An end-to-end retail analytics and machine learning project for forecasting product demand and generating data-driven inventory replenishment recommendations.

## Overview

This project combines Python, SQL, machine learning, and business analytics to build a retail demand forecasting and inventory optimization pipeline.

The system:

- Cleans and transforms historical retail sales data
- Performs exploratory and statistical analysis
- Stores analytical data in SQLite using SQL
- Engineers temporal and demand-based forecasting features
- Trains an XGBoost demand forecasting model
- Evaluates the model against a 7-day naive baseline
- Generates inventory recommendations using forecast demand, lead time, safety stock, and reorder points
- Provides an interactive Streamlit dashboard

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **SQL / SQLite**
- **XGBoost**
- **Scikit-learn**
- **Plotly**
- **Streamlit**

## Dataset

The project uses the **M5 Forecasting** retail sales dataset.

The analysis uses historical product-level sales together with calendar and selling-price information.

Large raw and processed datasets are intentionally excluded from the GitHub repository.

## Project Pipeline

```text
M5 Retail Data
      ↓
Data Cleaning & Preprocessing
      ↓
SQLite Database + SQL Analysis
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Chronological Train / Validation / Test Split
      ↓
XGBoost Demand Forecasting
      ↓
Model Evaluation
      ↓
Inventory Optimization
      ↓
Streamlit Dashboard
```

## Forecasting Model

The XGBoost forecasting model uses temporal and demand-based features including:

- 7, 14, and 28-day lag features
- 7, 14, and 28-day rolling means
- 7, 14, and 28-day rolling standard deviations
- Day of week
- Weekend indicator
- Month
- Week
- Year
- Day of month
- Demand change

A chronological train/validation/test split is used to preserve the time-series structure and reduce temporal leakage.

### Feature Importance

The most influential features in the trained model include:

1. `rolling_mean_7`
2. `rolling_mean_14`
3. `is_weekend`
4. `rolling_mean_28`
5. `lag_7`
6. `day_of_week`

## Model Performance

The XGBoost model was compared against a 7-day naive forecasting baseline.

| Metric | Result |
|---|---:|
| Naive 7-day MAE | 4.0843 |
| Naive 7-day RMSE | 6.8357 |
| XGBoost Validation MAE | 3.1097 |
| XGBoost Validation RMSE | 5.0860 |
| Validation WAPE | 48.54% |
| Validation MAE Improvement | **23.86%** |
| XGBoost Test MAE | **3.2651** |
| XGBoost Test RMSE | **5.1440** |
| XGBoost Test WAPE | **47.53%** |

The XGBoost model improved validation MAE by **23.86%** compared with the 7-day naive baseline.

## Inventory Optimization

The inventory layer converts demand forecasts into replenishment recommendations using:

- Forecasted daily demand
- Lead time
- Demand variability
- Safety stock
- Reorder points
- Inventory risk classification
- Recommended order quantity

The current test pipeline produces **7,200 forecast records** and corresponding inventory recommendations.

The framework is designed to translate machine learning forecasts into actionable inventory decisions.

## Dashboard

The Streamlit dashboard provides:

- Forecast demand KPIs
- Actual vs. predicted demand
- Inventory risk analysis
- Recommended order quantities
- Product-level recommendations
- Business-oriented analytics

Run the dashboard locally with:

```bash
streamlit run dashboard/app.py
```

Then open:

```text
http://localhost:8501
```
## Dashboard Preview

### Forecast Overview

![Forecast Dashboard](dashboard/dashboard%201.png)

### Demand Forecasting

![Demand Forecasting Dashboard](dashboard/dashboard%202.png)

### Inventory Optimization

![Inventory Optimization Dashboard](dashboard/dashboard%203.png)

## Project Structure

```text
retail-demand-forecasting-and-inventory-optimization/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── create_database.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── inventory_optimization.py
│   ├── preprocess.py
│   └── train_model.py
│
├── sql/
│   └── analysis.sql
│
├── download_data.py
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/saopreksha19-art/retail-demand-forecasting-and-inventory-optimization.git
cd retail-demand-forecasting-and-inventory-optimization
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python download_data.py
```

Run the project pipeline using the scripts in `src/`.

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Model Artifact

The trained XGBoost model is generated locally as:

```text
models/xgboost_retail_forecaster.json
```

The model artifact is excluded from GitHub because of its size. It can be regenerated by running:

```bash
python src/train_model.py
```

## Reproducibility

The repository contains the complete source code required to reproduce the analysis and modeling workflow.

Large datasets, generated processed files, the local virtual environment, and the trained model artifact are excluded through `.gitignore`.

## Key Takeaway

This project demonstrates an end-to-end workflow across:

**Data Analytics → SQL → Feature Engineering → Machine Learning → Forecast Evaluation → Inventory Optimization → Business Dashboard**

The project combines predictive modeling with operational decision-making rather than treating forecasting as an isolated machine learning task.

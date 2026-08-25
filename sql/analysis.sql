-- =========================================================
-- RETAIL DEMAND FORECASTING
-- SQL BUSINESS ANALYSIS
-- =========================================================


-- =========================================================
-- 1. Top products by average predicted demand
-- =========================================================

SELECT
    item_id,
    ROUND(AVG(predicted_demand), 2) AS avg_daily_forecast,
    ROUND(SUM(predicted_demand), 0) AS total_forecast
FROM forecast_predictions
GROUP BY item_id
ORDER BY avg_daily_forecast DESC
LIMIT 10;


-- =========================================================
-- 2. Top products requiring replenishment
-- =========================================================

SELECT
    item_id,
    ROUND(SUM(recommended_order_qty), 0)
        AS total_recommended_order
FROM inventory_recommendations
GROUP BY item_id
ORDER BY total_recommended_order DESC
LIMIT 10;


-- =========================================================
-- 3. Inventory risk distribution
-- =========================================================

SELECT
    inventory_risk,
    COUNT(*) AS observations,
    ROUND(
        100.0 * COUNT(*) /
        (SELECT COUNT(*)
         FROM inventory_recommendations),
        2
    ) AS percentage
FROM inventory_recommendations
GROUP BY inventory_risk
ORDER BY observations DESC;


-- =========================================================
-- 4. Products with highest safety-stock requirements
-- =========================================================

SELECT
    item_id,
    ROUND(AVG(demand_std), 2)
        AS avg_demand_variability,
    ROUND(AVG(safety_stock), 2)
        AS avg_safety_stock
FROM inventory_recommendations
GROUP BY item_id
ORDER BY avg_safety_stock DESC
LIMIT 10;


-- =========================================================
-- 5. Highest stockout-risk products
-- =========================================================

SELECT
    item_id,
    ROUND(
        AVG(days_until_stockout),
        2
    ) AS avg_days_until_stockout,
    ROUND(
        SUM(recommended_order_qty),
        0
    ) AS recommended_order_qty
FROM inventory_recommendations
GROUP BY item_id
ORDER BY avg_days_until_stockout ASC
LIMIT 10;


-- =========================================================
-- 6. Daily forecast vs actual demand
-- =========================================================

SELECT
    date,
    ROUND(SUM(sales), 0)
        AS actual_demand,
    ROUND(SUM(predicted_demand), 0)
        AS predicted_demand
FROM forecast_predictions
GROUP BY date
ORDER BY date;


-- =========================================================
-- 7. Forecast error by product
-- =========================================================

SELECT
    item_id,
    ROUND(
        AVG(ABS(sales - predicted_demand)),
        2
    ) AS mae
FROM forecast_predictions
GROUP BY item_id
ORDER BY mae DESC
LIMIT 10;


-- =========================================================
-- 8. Products with largest reorder quantities
-- =========================================================

SELECT
    item_id,
    ROUND(MAX(reorder_point), 0)
        AS max_reorder_point,
    ROUND(MAX(recommended_order_qty), 0)
        AS max_order_quantity
FROM inventory_recommendations
GROUP BY item_id
ORDER BY max_order_quantity DESC
LIMIT 10;

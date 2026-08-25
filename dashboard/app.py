import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import textwrap


# =========================================================
# CONFIGURATION
# =========================================================

DB_PATH = "data/processed/retail_forecasting.db"

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* -----------------------------------------------------
       GLOBAL
    ----------------------------------------------------- */

    .stApp {
        background: #0b1118;
        color: #e5e7eb;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    p {
        color: #94a3b8;
    }

    /* -----------------------------------------------------
       SIDEBAR
    ----------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #111b26;
        border-right: 1px solid #243244;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .sidebar-title {
        font-size: 21px;
        font-weight: 700;
        line-height: 1.25;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 24px;
        margin-bottom: 10px;
    }

    .sidebar-info {
        border-top: 1px solid #243244;
        padding-top: 18px;
        margin-top: 28px;
        font-size: 12px;
        color: #64748b;
        line-height: 1.7;
    }

    /* -----------------------------------------------------
       HEADER
    ----------------------------------------------------- */

    .dashboard-title {
        font-size: 31px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }

    .dashboard-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 20px;
    }

    .header-line {
        height: 1px;
        background: #243244;
        margin-bottom: 22px;
    }

       /* -----------------------------------------------------
       PANELS
    ----------------------------------------------------- */

    .panel {
        background: #111b26;
        border: 1px solid #263548;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 18px;
    }

    .panel-title {
        color: #f1f5f9;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .panel-subtitle {
        color: #64748b;
        font-size: 11px;
        margin-bottom: 12px;
    }

    /* -----------------------------------------------------
       TABLE
    ----------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #263548;
        border-radius: 8px;
    }

    /* -----------------------------------------------------
       SELECTBOX
    ----------------------------------------------------- */

    div[data-baseweb="select"] > div {
        background-color: #111b26;
        border-color: #334155;
    }

    /* -----------------------------------------------------
       BUTTON
    ----------------------------------------------------- */

    .stButton > button {
        width: 100%;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        height: 40px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #3b82f6;
        color: white;
    }

    /* -----------------------------------------------------
       FOOTER
    ----------------------------------------------------- */

    .footer {
        border-top: 1px solid #243244;
        margin-top: 35px;
        padding-top: 18px;
        text-align: center;
        color: #475569;
        font-size: 11px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATABASE LOADING
# =========================================================

@st.cache_data
def load_data():

    conn = sqlite3.connect(DB_PATH)

    forecast = pd.read_sql(
        "SELECT * FROM forecast_predictions",
        conn
    )

    inventory = pd.read_sql(
        "SELECT * FROM inventory_recommendations",
        conn
    )

    conn.close()

    forecast["date"] = pd.to_datetime(
        forecast["date"]
    )

    inventory["date"] = pd.to_datetime(
        inventory["date"]
    )

    return forecast, inventory


forecast, inventory = load_data()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            Retail Demand<br>
            Forecasting &<br>
            Inventory Optimization
        </div>

        <div class="sidebar-subtitle">
            Demand intelligence and replenishment planning
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Filters</div>',
        unsafe_allow_html=True
    )

    products = sorted(
        forecast["item_id"].dropna().unique()
    )

    selected_product = st.selectbox(
        "Product",
        ["All Products"] + products
    )

    stores = sorted(
        forecast["store_id"].dropna().unique()
    )

    selected_store = st.selectbox(
        "Store",
        ["All Stores"] + stores
    )

    min_date = forecast["date"].min().date()
    max_date = forecast["date"].max().date()

    selected_dates = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.markdown(
        '<div class="sidebar-section">Navigation</div>',
        unsafe_allow_html=True
    )

    st.radio(
        "Navigation",
        [
            "Overview",
            "Demand Forecast",
            "Inventory Optimization",
            "Product Analytics",
            "SQL Insights"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="sidebar-info">
            <strong>Data Source</strong><br>
            M5 Forecasting Dataset<br><br>

            <strong>Model</strong><br>
            XGBoost Demand Forecasting<br><br>

            <strong>Analytics</strong><br>
            Python + SQL + Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FILTER DATA
# =========================================================

filtered_forecast = forecast.copy()
filtered_inventory = inventory.copy()


if selected_product != "All Products":

    filtered_forecast = filtered_forecast[
        filtered_forecast["item_id"] == selected_product
    ]

    filtered_inventory = filtered_inventory[
        filtered_inventory["item_id"] == selected_product
    ]


if selected_store != "All Stores":

    filtered_forecast = filtered_forecast[
        filtered_forecast["store_id"] == selected_store
    ]

    filtered_inventory = filtered_inventory[
        filtered_inventory["store_id"] == selected_store
    ]


if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

    start_date = pd.Timestamp(selected_dates[0])
    end_date = pd.Timestamp(selected_dates[1])

    filtered_forecast = filtered_forecast[
        (filtered_forecast["date"] >= start_date)
        &
        (filtered_forecast["date"] <= end_date)
    ]

    filtered_inventory = filtered_inventory[
        (filtered_inventory["date"] >= start_date)
        &
        (filtered_inventory["date"] <= end_date)
    ]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="dashboard-title">
        Retail Demand Forecasting & Inventory Optimization
    </div>

    <div class="dashboard-subtitle">
        End-to-end analytics system combining demand forecasting,
        machine learning, SQL analytics and inventory optimization.
    </div>

    <div class="header-line"></div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_forecast = filtered_forecast[
    "predicted_demand"
].sum()

avg_daily = filtered_forecast[
    "predicted_demand"
].mean()

total_orders = filtered_inventory[
    "recommended_order_qty"
].sum()

medium_risk = (
    filtered_inventory["inventory_risk"]
    .eq("MEDIUM")
    .sum()
)

low_risk = (
    filtered_inventory["inventory_risk"]
    .eq("LOW")
    .sum()
)

high_risk = (
    filtered_inventory["inventory_risk"]
    .eq("HIGH")
    .sum()
)


# =========================================================
# KPI CARDS
# =========================================================

kpi_cols = st.columns(6)

kpi_data = [
    ("Total Forecast Demand", f"{total_forecast:,.0f}", "units"),
    ("Average Daily Forecast", f"{avg_daily:.2f}", "units"),
    ("Recommended Order Quantity", f"{total_orders:,.0f}", "units"),
    (
        "Products Needing Replenishment",
        f"{(filtered_inventory['recommended_order_qty'] > 0).sum():,}",
        "items"
    ),
    ("Low Risk Products", f"{low_risk:,}", "items"),
    ("High Risk Products", f"{high_risk:,}", "items"),
]

for col, (label, value, unit) in zip(kpi_cols, kpi_data):
    with col:
        st.metric(
            label=label,
            value=f"{value} {unit}"
        )

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# FORECAST DATA
# =========================================================


# =========================================================


# =========================================================

daily = (
    filtered_forecast
    .groupby("date", as_index=False)
    .agg(
        actual_demand=("sales", "sum"),
        predicted_demand=("predicted_demand", "sum")
    )
)


# =========================================================
# FORECAST CHART
# =========================================================

col_left, col_right = st.columns([1.65, 1])

with col_left:

    st.markdown(
        """
        <div class="panel-title">
            Actual vs Predicted Demand Over Time
        </div>

        <div class="panel-subtitle">
            Daily aggregated demand across the selected products and stores
        </div>
        """,
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["actual_demand"],
            mode="lines",
            name="Actual Demand",
            line=dict(
                color="#3b82f6",
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["predicted_demand"],
            mode="lines",
            name="Predicted Demand",
            line=dict(
                color="#67e8f9",
                width=2
            )
        )
    )

    fig.update_layout(
        height=340,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        paper_bgcolor="#111b26",
        plot_bgcolor="#111b26",
        font=dict(
            color="#94a3b8",
            size=11
        ),
        xaxis=dict(
            gridcolor="#243244",
            title="Date"
        ),
        yaxis=dict(
            gridcolor="#243244",
            title="Units"
        ),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0
        ),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False
        }
    )


# =========================================================
# RISK DONUT
# =========================================================

with col_right:

    st.markdown(
        """
        <div class="panel-title">
            Inventory Risk Distribution
        </div>

        <div class="panel-subtitle">
            Current replenishment risk across selected records
        </div>
        """,
        unsafe_allow_html=True
    )

    risk_data = (
        filtered_inventory[
            "inventory_risk"
        ]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH"],
            fill_value=0
        )
        .reset_index()
    )

    risk_data.columns = [
        "risk",
        "count"
    ]

    fig_risk = go.Figure(
        data=[
            go.Pie(
                labels=risk_data["risk"],
                values=risk_data["count"],
                hole=0.62,
                textinfo="label+percent",
                marker=dict(
                    colors=[
                        "#4ade80",
                        "#facc15",
                        "#ef4444"
                    ]
                )
            )
        ]
    )

    fig_risk.update_layout(
        height=340,
        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0
        ),
        paper_bgcolor="#111b26",
        font=dict(
            color="#cbd5e1"
        ),
        showlegend=True,
        legend=dict(
            orientation="v"
        )
    )

    st.plotly_chart(
    fig_risk,
    width="stretch",
        config={
            "displayModeBar": False
        }
    )


# =========================================================
# TOP REORDER PRODUCTS
# =========================================================

col_left, col_right = st.columns([1.65, 1])

with col_left:

    st.markdown(
        """
        <div class="panel-title">
            Top 10 Products by Recommended Order Quantity
        </div>

        <div class="panel-subtitle">
            Products with the largest replenishment requirements
        </div>
        """,
        unsafe_allow_html=True
    )

    top_orders = (
        filtered_inventory
        .groupby("item_id", as_index=False)
        .agg(
            recommended_order_qty=(
                "recommended_order_qty",
                "sum"
            ),
            avg_forecast=(
                "predicted_demand",
                "mean"
            )
        )
        .sort_values(
            "recommended_order_qty",
            ascending=False
        )
        .head(10)
    )

    fig_orders = px.bar(
        top_orders,
        x="recommended_order_qty",
        y="item_id",
        orientation="h",
        text="recommended_order_qty"
    )

    fig_orders.update_traces(
        marker_color="#60a5fa",
        textposition="outside"
    )

    fig_orders.update_layout(
        height=360,
        margin=dict(
            l=10,
            r=35,
            t=10,
            b=10
        ),
        paper_bgcolor="#111b26",
        plot_bgcolor="#111b26",
        font=dict(
            color="#94a3b8",
            size=10
        ),
        xaxis=dict(
            gridcolor="#243244",
            title="Recommended Order Quantity"
        ),
        yaxis=dict(
            gridcolor="#111b26",
            title=""
        ),
        showlegend=False
    )
    st.plotly_chart(
        fig_orders,
        width="stretch",
        config={
            "displayModeBar": False
        }
    )
# =========================================================
# SUMMARY STATISTICS
# =========================================================

with col_right:

    st.markdown(
        """
        <div class="panel-title">
            Summary Statistics
        </div>

        <div class="panel-subtitle">
            Key inventory planning metrics
        </div>
        """,
        unsafe_allow_html=True
    )

    avg_safety = filtered_inventory[
        "safety_stock"
    ].mean()

    avg_reorder = filtered_inventory[
        "reorder_point"
    ].mean()

    summary = pd.DataFrame(
        {
            "Metric": [
                "Average Predicted Daily Demand",
                "Average Safety Stock",
                "Average Reorder Point",
                "Total Recommended Order Quantity",
                "Total Number of Records",
                "Lead Time (Days)",
                "Service Level (Z)",
                "Inventory Coverage (Days)"
            ],
            "Value": [
                f"{avg_daily:.2f}",
                f"{avg_safety:.2f}",
                f"{avg_reorder:.2f}",
                f"{total_orders:,.0f}",
                f"{len(filtered_inventory):,}",
                "7",
                "1.645",
                "10"
            ]
        }
    )

    st.dataframe(
    summary,
    width="stretch",
    hide_index=True,
    height=350
)


# =========================================================
# INVENTORY RECOMMENDATIONS
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="panel-title">
        Latest Inventory Recommendations
    </div>

    <div class="panel-subtitle">
        Highest-priority replenishment recommendations generated by the inventory model
    </div>
    """,
    unsafe_allow_html=True
)

recommendation_columns = [
    "date",
    "item_id",
    "store_id",
    "predicted_demand",
    "safety_stock",
    "reorder_point",
    "estimated_inventory",
    "days_until_stockout",
    "recommended_order_qty",
    "inventory_risk"
]

recommendations = (
    filtered_inventory[
        recommendation_columns
    ]
    .sort_values(
        "recommended_order_qty",
        ascending=False
    )
    .head(15)
    .copy()
)

recommendations["date"] = (
    recommendations["date"]
    .dt.strftime("%Y-%m-%d")
)

recommendations.columns = [
    "Date",
    "Product ID",
    "Store ID",
    "Predicted Demand",
    "Safety Stock",
    "Reorder Point",
    "Estimated Inventory",
    "Days Until Stockout",
    "Recommended Order Qty",
    "Risk Level"
]
st.dataframe(
    recommendations,
    width="stretch",
    hide_index=True,
    height=430
)

# =========================================================
# PRODUCT ANALYTICS
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="panel-title">
        Product Analytics
    </div>

    <div class="panel-subtitle">
        Product-level demand and replenishment statistics
    </div>
    """,
    unsafe_allow_html=True
)

product_summary = (
    filtered_inventory
    .groupby("item_id", as_index=False)
    .agg(
        avg_forecast=(
            "predicted_demand",
            "mean"
        ),
        avg_safety_stock=(
            "safety_stock",
            "mean"
        ),
        avg_reorder_point=(
            "reorder_point",
            "mean"
        ),
        recommended_orders=(
            "recommended_order_qty",
            "sum"
        )
    )
    .sort_values(
        "recommended_orders",
        ascending=False
    )
    .head(20)
)

product_summary.columns = [
    "Product ID",
    "Avg Daily Forecast",
    "Avg Safety Stock",
    "Avg Reorder Point",
    "Recommended Order Qty"
]

st.dataframe(
    product_summary,
    width="stretch",
    hide_index=True,
    height=500
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Retail Demand Forecasting System
        &nbsp; | &nbsp;
        Python
        &nbsp; • &nbsp;
        XGBoost
        &nbsp; • &nbsp;
        SQL
        &nbsp; • &nbsp;
        Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

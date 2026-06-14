import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


st.set_page_config(
    page_title="NeuralStock",
    page_icon="📦",
    layout="wide"
)

st.title("📦 NeuralStock")
st.subheader("AI-Powered Inventory Demand Forecasting System")

DATA_PATH = "data/ecommerce_inventory_demand.csv"
MODEL_PATH = "models/lstm_inventory_model.pth"
X_SCALER_PATH = "models/x_scaler.pkl"
Y_SCALER_PATH = "models/y_scaler.pkl"
CATEGORY_ENCODER_PATH = "models/category_encoder.pkl"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

with open(X_SCALER_PATH, "rb") as f:
    x_scaler = pickle.load(f)

with open(Y_SCALER_PATH, "rb") as f:
    y_scaler = pickle.load(f)

with open(CATEGORY_ENCODER_PATH, "rb") as f:
    category_encoder = pickle.load(f)

model = LSTMModel(input_size=11)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

st.subheader("📊 Business KPI Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Products", df["product_id"].nunique())
col2.metric("Categories", df["product_category"].nunique())
col3.metric("Avg Units Sold", round(df["units_sold"].mean(), 2))
col4.metric("Avg Stock", round(df["stock_on_hand"].mean(), 2))

st.subheader("📋 Dataset Preview")
st.dataframe(df.head(20))

st.subheader("📈 Demand Trend Analysis")

daily_sales = df.groupby("date")["units_sold"].sum().reset_index()

fig = px.line(
    daily_sales,
    x="date",
    y="units_sold",
    title="Daily Demand Trend"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Category Performance")

category_sales = (
    df.groupby("product_category")["units_sold"]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    category_sales,
    x="product_category",
    y="units_sold",
    title="Average Sales by Category"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("🏆 Top Selling Products")

top_products = (
    df.groupby("product_id")["units_sold"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig3 = px.bar(
    top_products,
    x="product_id",
    y="units_sold",
    title="Top 10 Products by Total Sales"
)

st.plotly_chart(fig3, use_container_width=True)

st.subheader("🎯 Promotion Impact")

promo_sales = (
    df.groupby("is_promotion")["units_sold"]
    .mean()
    .reset_index()
)

promo_sales["is_promotion"] = promo_sales["is_promotion"].map(
    {0: "No Promotion", 1: "Promotion"}
)

fig4 = px.bar(
    promo_sales,
    x="is_promotion",
    y="units_sold",
    title="Average Sales by Promotion Status"
)

st.plotly_chart(fig4, use_container_width=True)

st.subheader("🚨 Inventory Alerts")

low_stock = df[df["stock_on_hand"] < df["reorder_point"]]

st.metric("Products Requiring Reorder", len(low_stock))

st.dataframe(
    low_stock[
        [
            "product_id",
            "product_category",
            "stock_on_hand",
            "reorder_point"
        ]
    ].head(20)
)

st.subheader("🧠 AI Demand Prediction")

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox(
        "Product Category",
        sorted(df["product_category"].unique())
    )

    stock_on_hand = st.number_input(
        "Stock On Hand",
        min_value=0,
        value=100
    )

    reorder_point = st.number_input(
        "Reorder Point",
        min_value=0,
        value=30
    )

    supplier_lead_days = st.number_input(
        "Supplier Lead Days",
        min_value=1,
        value=7
    )

with col2:
    promotion_text = st.selectbox(
        "Promotion",
        ["No", "Yes"]
    )

    discount_pct = st.slider(
        "Discount %",
        0,
        50,
        10
    )

    weekend_text = st.selectbox(
        "Weekend",
        ["No", "Yes"]
    )

is_promotion = 1 if promotion_text == "Yes" else 0
is_weekend = 1 if weekend_text == "Yes" else 0
category_encoded = category_encoder.transform([category])[0]

category_data = df[df["product_category"] == category]

lag_7 = category_data["units_sold"].median()
lag_14 = category_data["units_sold"].median()
rolling_mean_7 = category_data["units_sold"].rolling(7).mean().dropna().median()
rolling_mean_30 = category_data["units_sold"].rolling(30).mean().dropna().median()

if st.button("Predict Demand"):
    input_data = np.array([[
        stock_on_hand,
        reorder_point,
        is_promotion,
        discount_pct,
        supplier_lead_days,
        is_weekend,
        lag_7,
        lag_14,
        rolling_mean_7,
        rolling_mean_30,
        category_encoded
    ]])

    input_scaled = x_scaler.transform(input_data)

    input_tensor = torch.tensor(
        input_scaled.reshape(1, 1, 11),
        dtype=torch.float32
    )

    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    prediction = y_scaler.inverse_transform(
        prediction_scaled.numpy()
    )[0][0]

    predicted_units = max(0, round(prediction))

    st.success(f"Predicted Demand: {predicted_units} units")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted_units,
            title={"text": "Predicted Demand"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "lightblue"},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 70], "color": "orange"},
                    {"range": [70, 100], "color": "red"}
                ],
            },
        )
    )

    st.plotly_chart(gauge, use_container_width=True)

    if predicted_units > stock_on_hand:
        st.error("🔴 Stockout Risk: Demand is higher than available stock.")
    elif stock_on_hand <= reorder_point:
        st.warning("🟡 Reorder Recommended: Stock is near or below reorder point.")
    else:
        st.info("🟢 Stock Level Healthy")
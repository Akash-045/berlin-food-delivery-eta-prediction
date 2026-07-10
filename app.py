import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt

# ============================================================
# BRAND PALETTE (matches presentation deck)
# ============================================================
NAVY = "#2F3C7E"
CORAL = "#F96167"
GOLD = "#F9E795"
LIGHT_CORAL = "#FDEBEA"
LIGHT_NAVY = "#E7E9F5"
OFFWHITE = "#FAFAFA"
DARKTEXT = "#23283A"
MUTED = "#6B7280"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Berlin Food Delivery ETA Predictor",
    page_icon="🛵",
    layout="centered"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(f"""
<style>
    .stApp {{
        background-color: {OFFWHITE};
    }}
    h1, h2, h3 {{
        color: {NAVY} !important;
        font-family: Georgia, serif;
    }}
    .stButton > button {{
        background-color: {CORAL};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6em 1.2em;
    }}
    .stButton > button:hover {{
        background-color: {NAVY};
        color: white;
    }}
    div[data-testid="stMetricValue"] {{
        color: {NAVY};
    }}
    .footer-box {{
        background-color: {NAVY};
        padding: 1.4rem;
        border-radius: 12px;
        margin-top: 1.5rem;
    }}
    .footer-box a {{
        color: {GOLD} !important;
        text-decoration: none;
        font-weight: 600;
    }}
    .platform-badge {{
        display: inline-block;
        background-color: {LIGHT_CORAL};
        color: {NAVY};
        padding: 0.25em 0.7em;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 0.2em;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    model_path = "delivery_eta_model.pkl"
    cols_path = "feature_columns.json"
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        return None, None
    model = joblib.load(model_path)
    with open(cols_path) as f:
        feature_columns = json.load(f)
    return model, feature_columns

model, feature_columns = load_model()

# ============================================================
# HAVERSINE DISTANCE (same formula used in training)
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ============================================================
# HEADER
# ============================================================
st.title("🛵 Berlin Food Delivery ETA Predictor")
st.markdown(
    "Predict how long a food delivery will take, based on a machine learning model "
    "trained on distance, traffic, weather, and rider context."
)
st.markdown(
    f"""
    <span class="platform-badge">🛵 Inspired by Wolt</span>
    <span class="platform-badge">🍔 Uber Eats</span>
    <span class="platform-badge">🥡 Lieferando</span>
    """,
    unsafe_allow_html=True
)
st.caption("*Built by a former Wolt delivery rider, as a Machine Learnining Project.*")
st.divider()

if model is None:
    st.error(
        "⚠️ Model files not found. Please make sure `delivery_eta_model.pkl` and "
        "`feature_columns.json` are in the same folder as this app."
    )
    st.stop()

# ============================================================
# INPUT FORM
# ============================================================
st.subheader("📍 Delivery Details")

col1, col2 = st.columns(2)
with col1:
    distance_km = st.slider("Distance (km)", 0.5, 30.0, 5.0, 0.1)
with col2:
    traffic_level = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"], index=1)

col3, col4 = st.columns(2)
with col3:
    weather = st.selectbox(
        "Weather Condition",
        ["Sunny", "Windy", "Stormy", "Sandstorms", "Cloudy", "Fog"],
        index=0
    )
with col4:
    vehicle_type = st.selectbox(
        "Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"], index=1
    )

st.subheader("🕐 Order Timing")
col5, col6 = st.columns(2)
with col5:
    order_time = st.time_input("Order time", value=datetime.now().time())
with col6:
    order_date = st.date_input("Order date", value=datetime.now().date())

prep_time_min = st.slider(
    "Restaurant's estimated prep time (minutes)", 5, 30, 12,
    help="How long the restaurant expects to take preparing the order before pickup."
)

st.subheader("🧑 Rider & Order Context")
col7, col8 = st.columns(2)
with col7:
    delivery_person_age = st.slider("Delivery person age", 18, 60, 30)
with col8:
    delivery_person_rating = st.slider("Delivery person rating", 1.0, 5.0, 4.6, 0.1)

col9, col10 = st.columns(2)
with col9:
    vehicle_condition = st.selectbox("Vehicle condition (0=poor, 3=excellent)", [0, 1, 2, 3], index=2)
with col10:
    multiple_deliveries = st.selectbox("Other orders bundled with this one", [0, 1, 2, 3], index=0)

col11, col12 = st.columns(2)
with col11:
    order_type = st.selectbox("Order type", ["Snack", "Meal", "Drinks", "Buffet"], index=1)
with col12:
    city_type = st.selectbox("City type", ["Urban", "Metropolitian", "Semi-Urban"], index=1)

festival = st.radio("Is today a festival day?", ["No", "Yes"], horizontal=True)

st.divider()

# ============================================================
# BUILD FEATURE VECTOR
# ============================================================
def build_input_row():
    order_hour = order_time.hour
    day_of_week = order_date.weekday()
    is_weekend = int(day_of_week in [5, 6])
    is_rush_hour = int(order_hour in [12, 13, 19, 20])

    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    traffic_level_num = traffic_map[traffic_level]
    distance_x_traffic = distance_km * traffic_level_num
    is_bad_weather = int(weather in ["Stormy", "Sandstorms", "Fog"])

    row = {
        "Delivery_person_Age": delivery_person_age,
        "Delivery_person_Ratings": delivery_person_rating,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries,
        "distance_km": distance_km,
        "order_hour": order_hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "prep_time_min": prep_time_min,
        "traffic_level_num": traffic_level_num,
        "distance_x_traffic": distance_x_traffic,
        "is_bad_weather": is_bad_weather,
        # Berlin weather: use seasonal-average placeholders since live values
        # weren't available at inference time in this simplified demo
        "berlin_temp_c": 10.0,
        "berlin_precip_mm": 0.5,
        "berlin_windspeed_max": 18.0,
    }

    # One-hot encode categorical fields to match training columns exactly
    for col in feature_columns:
        if col.startswith("Type_of_order_"):
            row[col] = int(col == f"Type_of_order_{order_type}")
        elif col.startswith("Type_of_vehicle_"):
            row[col] = int(col == f"Type_of_vehicle_{vehicle_type}")
        elif col.startswith("City_"):
            row[col] = int(col == f"City_{city_type}")
        elif col.startswith("Festival_"):
            row[col] = int(col == f"Festival_{festival}")

    # Fill any remaining expected columns with 0, then reorder to match training
    input_df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    return input_df

# ============================================================
# PREDICT
# ============================================================
if st.button("🔮 Predict Delivery Time", type="primary", use_container_width=True):
    input_df = build_input_row()
    prediction = model.predict(input_df)[0]
    prediction = max(prediction, 5)  # sanity floor

    st.success(f"### Estimated delivery time: **{prediction:.0f} minutes**")

    low, high = prediction - 4.2, prediction + 4.2
    st.caption(f"Typical range: {low:.0f}–{high:.0f} minutes (based on model's average error of ±4.2 min)")

    with st.expander("See what influenced this prediction"):
        st.write(f"- **Distance:** {distance_km} km in **{traffic_level}** traffic")
        st.write(f"- **Weather:** {weather}")
        st.write(f"- **Vehicle:** {vehicle_type}")
        st.write(f"- **Rider rating:** {delivery_person_rating} ⭐")
        st.write(f"- **Bundled orders:** {multiple_deliveries}")
        st.write(f"- **Restaurant prep time:** {prep_time_min} min")

st.divider()

# ============================================================
# MODEL INSIGHTS: FEATURE IMPORTANCE
# ============================================================
st.subheader("📊 What Drives This Model's Predictions?")
st.caption("Feature importance pulled directly from the trained model — not hardcoded.")

importances = pd.Series(model.feature_importances_, index=feature_columns)
top_features = importances.sort_values(ascending=True).tail(8)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(top_features.index, top_features.values, color=CORAL)
ax.set_facecolor(OFFWHITE)
fig.patch.set_facecolor(OFFWHITE)
ax.set_xlabel("Importance", color=DARKTEXT, fontsize=10)
ax.tick_params(colors=DARKTEXT, labelsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#D1D5DB")
ax.set_title("Top 8 Features (Gradient Boosting)", color=NAVY, fontsize=12, fontweight="bold", loc="left")
plt.tight_layout()
st.pyplot(fig)

st.caption(
    "Delivery person rating and the distance × traffic interaction consistently rank highest — "
    "confirming that rider experience and congestion together matter more than distance alone."
)

st.divider()

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f"""
    <div class="footer-box">
        <div style="color: white; font-size: 1.05em; font-weight: 700;">Akash Samantray</div>
        <div style="color: {GOLD}; font-size: 0.85em; margin-bottom: 0.6em;">
            Data Analyst · Former Wolt Delivery Rider · Berlin, Germany
        </div>
        <div style="color: #C7CCEA; font-size: 0.85em; line-height: 1.6;">
            Model: Tuned Gradient Boosting Regressor (R² ≈ 0.68, MAE ≈ 4.2 min)<br>
            Trained on 45,000+ food delivery orders · Ironhack Berlin ML Project<br>
            <a href="https://www.linkedin.com/in/akash-samantray/" target="_blank">🔗 Connect on LinkedIn</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
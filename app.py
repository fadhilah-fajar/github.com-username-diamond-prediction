# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# 1. Load model (dan scaler jika KNN)
# ================================
best_model = joblib.load("deploy_diamond.pkl")

# Cek apakah model KNN
try:
    scaler = joblib.load("scaler_knn.pkl")
    is_knn = True
except:
    scaler = None
    is_knn = False

st.title("Diamond Price Prediction 💎")
st.write("Prediksi harga diamond berdasarkan fitur yang dimasukkan.")

# ================================
# 2. Input fitur dari user
# ================================
st.sidebar.header("Input Features")

# Fitur numerik
carat = st.sidebar.number_input("Carat", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
depth = st.sidebar.number_input("Depth (%)", min_value=50.0, max_value=75.0, value=60.0, step=0.1)
table = st.sidebar.number_input("Table (%)", min_value=50.0, max_value=100.0, value=57.0, step=0.1)
x = st.sidebar.number_input("Length (x)", min_value=0.0, max_value=10.0, value=5.0, step=0.01)
y = st.sidebar.number_input("Width (y)", min_value=0.0, max_value=10.0, value=5.0, step=0.01)
z = st.sidebar.number_input("Depth (z)", min_value=0.0, max_value=10.0, value=3.0, step=0.01)

# Fitur kategori
cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.sidebar.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.sidebar.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

# ================================
# 3. Prepare input dataframe
# ================================
input_dict = {
    "carat": carat,
    "depth": depth,
    "table": table,
    "x": x,
    "y": y,
    "z": z
}

input_df = pd.DataFrame([input_dict])

# Encoding kategori sama seperti di training
# Pastikan drop_first=True
for col, categories in zip(
    ["cut","color","clarity"],
    [
        ["Good","Ideal","Premium","Very Good"],  # cut
        ["E","F","G","H","I","J"],              # color
        ["IF","SI1","SI2","VS1","VS2","VVS1","VVS2"]  # clarity
    ]
):
    for cat in categories:
        col_name = f"{col}_{cat}"
        input_df[col_name] = 1 if eval(col.lower()) == cat else 0

# ================================
# 4. Scaling jika KNN
# ================================
if is_knn:
    input_scaled = scaler.transform(input_df)
    prediction = best_model.predict(input_scaled)[0]
else:
    prediction = best_model.predict(input_df)[0]

# ================================
# 5. Display Prediction
# ================================
st.subheader("Predicted Diamond Price 💰")
st.write(f"${prediction:,.2f}")
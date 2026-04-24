import streamlit as st
import numpy as np
import pickle

# Load model + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="MPG Predictor", layout="wide")

# Background CSS (Car theme + dark glass effect)
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1503376780353-7e6692767b70");
    background-size: cover;
    background-attachment: fixed;
}

.main {
    background: rgba(0,0,0,0.65);
    padding: 20px;
    border-radius: 20px;
    color: white;
}

h1 {
    text-align: center;
    color: #00d4ff;
    font-size: 40px;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px;
}
label {
    color: #4169E1 !important;   
    font-weight: 600 !important;
    font-size: 35px !important;
}
.stNumberInput input {
    background-color: #1e1e1e !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid #555 !important;
}
.stNumberInput input:focus {
    border: 1px solid #FFD700 !important;
    box-shadow: 0px 0px 5px #FFD700 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚗 MPG PREDICTION DASHBOARD")

st.write("Enter car specifications below to predict fuel efficiency")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    cylinders = st.number_input("Cylinders", 3, 12, 6)
    displacement = st.number_input("Displacement", 0.0, 500.0, 150.0)
    horsepower = st.number_input("Horsepower", 0.0, 300.0, 90.0)
    weight = st.number_input("Weight", 0.0, 6000.0, 3000.0)

with col2:
    acceleration = st.number_input("Acceleration", 0.0, 30.0, 10.0)
    model_year = st.number_input("Model Year", 70, 100, 85)
    origin = st.number_input("Origin", 1, 3, 1)

# ---------------- PREDICT ----------------
if st.button("🚀 Predict MPG"):

    input_data = np.array([[cylinders, displacement, horsepower, weight,
                            acceleration, model_year, origin]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"🚗 Predicted MPG: {prediction[0]:.2f}")

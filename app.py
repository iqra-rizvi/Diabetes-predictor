import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")

st.markdown("Enter the following patient details:")

# Input fields
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction_proba = model.predict_proba(input_scaled)[0]
    prediction = np.argmax(prediction_proba)
    confidence = round(prediction_proba[prediction] * 100, 2)

    st.subheader("Result:")
    if prediction == 1:
        st.error(f"⚠️ Likely Diabetic (Confidence: {confidence}%)")
    else:
        st.success(f"✅ Not Diabetic (Confidence: {confidence}%)")

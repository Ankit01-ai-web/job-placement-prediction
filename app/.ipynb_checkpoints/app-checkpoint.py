import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('../model/placement_model.pkl')

st.title("🎓 Job Placement Prediction App")
st.markdown("Predict if a student will be placed based on their academic profile.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
ssc_p = st.slider("Secondary Education (SSC) Percentage", 40.0, 100.0, 75.0)
ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
hsc_p = st.slider("Higher Secondary (HSC) Percentage", 40.0, 100.0, 75.0)
hsc_b = st.selectbox("HSC Board", ["CBSE", "ICSE", "OPEN", "Others"])
hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])
degree_p = st.slider("Degree Percentage", 40.0, 100.0, 70.0)
degree_t = st.selectbox("Undergrad Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
workex = st.selectbox("Work Experience", ["Yes", "No"])
etest_p = st.slider("Employability Test Percentage", 0.0, 100.0, 70.0)
specialisation = st.selectbox("Specialisation", ["Computer Application", "AI & DS", "Machine Learning"])
mca_p = st.slider("MCA Percentage", 40.0, 100.0, 70.0)

# Encode user inputs
def encode_input():
    input_data = [
        1 if gender == "Male" else 0,
        ssc_p,
        1 if ssc_b == "Central" else 0,
        hsc_p,
         {"CBSE": 0, "ICSE": 1, "OPEN": 2}[hsc_b],
        {"Commerce": 0, "Science": 1, "Arts": 2}[hsc_s],
        degree_p,
        {"Sci&Tech": 0, "Comm&Mgmt": 1, "Others": 2}[degree_t],
        1 if workex == "Yes" else 0,
        etest_p,
        {"Computer Application": 0, "AI & DS": 1, "Machine Learning": 2}[specialisation],
        mca_p
    ]
    return np.array([input_data])

# Predict
if st.button("Predict Placement"):
    input_array = encode_input()
    prediction = model.predict(input_array)
    result = "✅ Placed" if prediction[0] == 1 else "❌ Not Placed"
    st.subheader(f"Prediction: {result}")

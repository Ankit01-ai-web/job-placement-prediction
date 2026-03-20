import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load('model/model.pkl')

st.set_page_config(page_title="Placement Predictor", layout="centered")

st.title("🎓 Job Placement Prediction System")
st.markdown("Predict whether a student will be placed or not using ML")


# SINGLE PREDICTION

st.header("🔍 Single Prediction")

gender = st.selectbox("Gender", ["M", "F"])
ssc_p = st.slider("SSC Percentage", 40.0, 100.0, 70.0)
ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
hsc_p = st.slider("HSC Percentage", 40.0, 100.0, 70.0)
hsc_b = st.selectbox("HSC Board", ["CBSE", "ICSE", "Others"])
hsc_s = st.selectbox("Stream", ["Commerce", "Science", "Arts"])
degree_p = st.slider("Degree Percentage", 40.0, 100.0, 70.0)
degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
workex = st.selectbox("Work Experience", ["Yes", "No"])
etest_p = st.slider("Employability Test %", 0.0, 100.0, 70.0)
specialisation = st.selectbox("Specialisation", ["Mkt&HR", "Mkt&Fin"])
mca_p = st.slider("MCA Percentage", 40.0, 100.0, 70.0)

# Convert to dataframe
input_df = pd.DataFrame([{
    "gender": gender,
    "ssc_p": ssc_p,
    "ssc_b": ssc_b,
    "hsc_p": hsc_p,
    "hsc_b": hsc_b,
    "hsc_s": hsc_s,
    "degree_p": degree_p,
    "degree_t": degree_t,
    "workex": workex,
    "etest_p": etest_p,
    "specialisation": specialisation,
    "mca_p": mca_p
}])

if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "Placed" if prediction[0] == "Placed" else "Not Placed"
    st.success(result)


# CSV BATCH PREDICTION

st.header("Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    try:
        predictions = model.predict(df)
        df["Prediction"] = predictions
        
        st.write(" Results:")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Download Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error: {e}")
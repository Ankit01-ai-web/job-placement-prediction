# Job Placement Prediction System

# Overview
This project is a Machine Learning-based system that predicts whether a student will be placed or not based on academic performance, skills, and other factors.  
It uses a trained model and provides predictions through a user-friendly web interface.

# Features
-  Machine Learning Classification Model
-  Data Preprocessing (Encoding + Scaling)
-  Pipeline Implementation (ColumnTransformer + Pipeline)
-  Multiple Models (Logistic Regression & Random Forest)
-  Hyperparameter Tuning (GridSearchCV)
-  Streamlit Web Application
-  Real-time Prediction
-  Batch Prediction using CSV Upload

# Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib


# How It Works
1. Dataset is loaded from data/Placement_Data.csv
2. Data is preprocessed using encoding and scaling
3. Model is trained using ML algorithms
4. Best model is saved in model/model.pkl
5. Streamlit app loads the model and takes user input
6. Prediction is displayed (Placed / Not Placed)


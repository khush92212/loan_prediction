
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("loan_prediction_model.pkl")

# Initialize the LabelEncoder for categorical features
label_encoders = {
    "Gender": LabelEncoder(),
    "Married": LabelEncoder(),
    "Dependents": LabelEncoder(),
    "Education": LabelEncoder(),
    "Self_Employed": LabelEncoder(),
    "Credit_History": LabelEncoder(),
    "Property_Area": LabelEncoder()
}

# Function to encode categorical features
def encode_categorical_features(df):
    for column in label_encoders:
        df[column] = label_encoders[column].fit_transform(df[column])
    return df

st.title("Loan Prediction App")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
married = st.selectbox("Married", ["Yes", "No"], key="married")
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], key="dependents")
education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="education")
self_employed = st.selectbox("Self Employed", ["Yes", "No"], key="self_employed")

# Numeric inputs with unique keys to prevent duplicate element ID errors
applicant_income = st.number_input("Applicant Income", min_value=0, step=1, value=0, key="applicant_income")
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=1, value=0, key="coapplicant_income")
loan_amount = st.number_input("Loan Amount", min_value=0, step=1, value=0, key="loan_amount")
loan_term = st.number_input("Loan Amount Term (in months)", min_value=0, step=1, value=0, key="loan_term")

credit_history = st.selectbox("Credit History", [1.0, 0.0], key="credit_history")
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], key="property_area")

# Create DataFrame for model input (Exclude Loan_ID and Loan_Status)
df = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})

# Ensure the correct feature order (based on model training)
expected_columns = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area"
]

df = df[expected_columns]

# Encode categorical features
df = encode_categorical_features(df)

# Make Prediction when button is clicked
if st.button("Predict"):
    try:
        prediction = model.predict(df)

        # Display result based on prediction
        if prediction[0] == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")
    except Exception as e:
        st.error(f"Error: {e}")

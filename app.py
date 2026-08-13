import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("loan_eligibility_random_forest.pkl")
feature_columns = joblib.load("loan_feature_columns.pkl")


def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    credit_history,
    property_area
):
    customer = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Loan_Amount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })

    categorical_columns = customer.select_dtypes(include=["object", "string"]).columns
    customer_encoded = pd.get_dummies(customer, columns=categorical_columns, drop_first=True)
    customer_encoded = customer_encoded.reindex(columns=feature_columns, fill_value=0).astype(int)

    prediction = model.predict(customer_encoded)[0]
    probability = model.predict_proba(customer_encoded)[0][1]

    if prediction == 1:
        status = "Loan Approved"
    else:
        status = "Loan Not Approved"

    return status, probability


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Eligibility Prediction System")
st.write("Enter the applicant details below to predict loan eligibility.")

st.divider()

# Input layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=150, step=10)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=1, value=360, step=12)
    credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Poor (0)")

st.divider()

if st.button("🔍 Predict Loan Eligibility", use_container_width=True):
    status, probability = predict_loan(
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount, loan_amount_term,
        credit_history, property_area
    )

    st.subheader("Prediction Result")
    if status == "Loan Approved":
        st.success(f"✅ {status}")
    else:
        st.error(f"❌ {status}")

    st.metric("Approval Probability", f"{probability * 100:.2f}%")
    st.progress(float(probability))

st.divider()
st.caption("Loan Eligibility Prediction System | Machine Learning Project")

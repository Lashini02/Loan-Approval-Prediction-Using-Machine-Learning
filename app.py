import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Loan Eligibility Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        padding: 1.5rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .approved-card {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        text-align: center;
        margin-bottom: 20px;
    }
    .rejected-card {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        text-align: center;
        margin-bottom: 20px;
    }
    .header-style {
        color: #1E3A8A;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model_path = "loan_eligibility_random_forest.pkl"
    features_path = "loan_feature_columns.pkl"
    
    model = None
    feature_columns = None
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(features_path):
        feature_columns = joblib.load(features_path)
        
    return model, feature_columns

model, feature_columns = load_artifacts()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=70)
    st.title("Loan Predictor AI")
    st.write("**Member 5 - Model Deployment**")
    st.markdown("---")
    
    st.subheader("📌 System Status")
    if model is not None and feature_columns is not None:
        st.success("✅ ML Model & Pipeline Loaded")
        st.info(f"Features Model Trained On: **{len(feature_columns)}**")
    else:
        st.error("⚠️ Model files not found! Please run training notebook first.")

    st.markdown("---")
    st.subheader("💡 Quick Sample Profiles")
    st.write("Click to pre-fill applicant data:")
    
    col_p1, col_p2 = st.columns(2)
    sample_choice = None
    if col_p1.button("👤 Profile A (Strong)"):
        sample_choice = "Strong"
    if col_p2.button("⚠️ Profile B (Risky)"):
        sample_choice = "Risky"
        
    st.markdown("---")
    st.caption("Powered by Machine Learning & Streamlit | Project Group Member 5")


# Main Page Header
st.markdown("<h1 class='header-style'>🏦 Loan Approval & Eligibility Prediction</h1>", unsafe_allow_html=True)
st.markdown("Enter the applicant's demographic, financial, and credit details below to generate real-time loan approval prediction and risk analysis.")
st.divider()

# Defaults handling for Quick Profiles
if sample_choice == "Strong":
    def_gender = "Male"
    def_married = "Yes"
    def_dependents = "0"
    def_education = "Graduate"
    def_self_emp = "No"
    def_app_inc = 6000
    def_coapp_inc = 2500
    def_loan_amt = 150
    def_term = 360
    def_credit = 1
    def_prop = "Semiurban"
elif sample_choice == "Risky":
    def_gender = "Female"
    def_married = "No"
    def_dependents = "2"
    def_education = "Not Graduate"
    def_self_emp = "Yes"
    def_app_inc = 2000
    def_coapp_inc = 0
    def_loan_amt = 250
    def_term = 360
    def_credit = 0
    def_prop = "Rural"
else:
    def_gender = "Male"
    def_married = "Yes"
    def_dependents = "0"
    def_education = "Graduate"
    def_self_emp = "No"
    def_app_inc = 5000
    def_coapp_inc = 1500
    def_loan_amt = 140
    def_term = 360
    def_credit = 1
    def_prop = "Urban"

# Input Form Sections
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Applicant Demographics")
    
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if def_gender == "Male" else 1)
        married = st.selectbox("Marital Status", ["Yes", "No"], index=0 if def_married == "Yes" else 1)
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=["0","1","2","3+"].index(def_dependents))
    with c2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if def_education == "Graduate" else 1)
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0 if def_self_emp == "No" else 1)
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban","Semiurban","Rural"].index(def_prop))

with col2:
    st.subheader("💰 Financial & Credit Profile")
    
    f1, f2 = st.columns(2)
    with f1:
        applicant_income = st.number_input("Applicant Income ($/month)", min_value=0, value=def_app_inc, step=500)
        coapplicant_income = st.number_input("Co-applicant Income ($/month)", min_value=0, value=def_coapp_inc, step=500)
        credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good Credit (1)" if x == 1 else "Poor / No Credit (0)", index=0 if def_credit == 1 else 1)
    with f2:
        loan_amount = st.number_input("Loan Amount (in Thousands $)", min_value=1, value=def_loan_amt, step=10)
        loan_amount_term = st.number_input("Loan Term (in Days)", min_value=12, value=def_term, step=12)

# Quick Derived Financial Summaries
st.markdown("<br>", unsafe_allow_html=True)
sc1, sc2, sc3 = st.columns(3)
total_income = applicant_income + coapplicant_income
sc1.metric("Combined Monthly Income", f"${total_income:,}")
sc2.metric("Requested Loan Amount", f"${loan_amount * 1000:,}")
est_term_months = max(1, loan_amount_term // 30)
sc3.metric("Estimated Term", f"{est_term_months} Months")

st.divider()

# Prediction Engine
def predict_eligibility(gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area):
    if model is None or feature_columns is None:
        return "Model Error", 0.0

    # Build input dataframe matching model schema
    input_dict = {
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
    }
    
    df_customer = pd.DataFrame(input_dict)
    
    # One-hot encoding
    cat_cols = df_customer.select_dtypes(include=["object"]).columns
    df_encoded = pd.get_dummies(df_customer, columns=cat_cols, drop_first=True)
    
    # Reindex to align with trained features
    df_final = df_encoded.reindex(columns=feature_columns, fill_value=0).astype(int)
    
    prediction = model.predict(df_final)[0]
    probabilities = model.predict_proba(df_final)[0]
    
    approval_prob = probabilities[1] if len(probabilities) > 1 else float(prediction)
    status = "Approved" if prediction == 1 else "Rejected"
    
    return status, approval_prob

# Predict Button
predict_btn = st.button("🚀 Evaluate Loan Eligibility", use_container_width=True, type="primary")

if predict_btn:
    status, prob = predict_eligibility(
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area
    )
    
    st.markdown("### 📊 Prediction Assessment Results")
    
    res_col1, res_col2 = st.columns([1.2, 1])
    
    with res_col1:
        if status == "Approved":
            st.markdown(f"""
                <div class='approved-card'>
                    <h2>🎉 LOAN APPROVED</h2>
                    <p>The applicant meets the eligibility criteria for loan approval.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='rejected-card'>
                    <h2>❌ LOAN NOT APPROVED</h2>
                    <p>The application presents elevated financial risk based on historical data.</p>
                </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        st.metric("Approval Probability Score", f"{prob * 100:.1f}%")
        st.progress(float(prob))
        
    # Risk Factor Breakdown
    with st.expander("🔍 Risk & Key Factor Analysis", expanded=True):
        risk_factors = []
        if credit_history == 0:
            risk_factors.append("⚠️ **Poor Credit History**: Credit history is the strongest predictor. Applicants with 0 credit history have significantly lower approval rates.")
        if total_income < 3000 and loan_amount > 150:
            risk_factors.append("⚠️ **High Loan-to-Income Ratio**: Requested loan amount is high relative to combined income.")
        if education == "Not Graduate":
            risk_factors.append("ℹ️ **Education Factor**: Non-graduate applications carry slightly higher risk weighting in historical data.")
            
        if not risk_factors:
            st.success("✅ Strong financial profile with positive credit history and adequate income coverage.")
        else:
            for factor in risk_factors:
                st.markdown(factor)

st.markdown("---")
st.caption("Loan Approval Prediction System | Developed for Machine Learning Coursework | Member 5: Model Deployment & UI Web Application")

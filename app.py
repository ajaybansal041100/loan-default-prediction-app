import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="AI Credit Risk Engine",
    page_icon="💳",
    layout="wide"
)

# ------------------------------------------------
# Custom Modern CSS
# ------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}
.main {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
.title {
    font-size:45px;
    font-weight:700;
    color:white;
}
.subtitle {
    color:#cfd8dc;
    font-size:18px;
}
.glass {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    font-size:18px;
    border-radius:12px;
    height:3em;
    width:100%;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load Model
# ------------------------------------------------

model = xgb.XGBClassifier()
model.load_model("model.json")

# ------------------------------------------------
# Header
# ------------------------------------------------
st.markdown('<div class="title">💳 AI Credit Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Loan Default Probability Prediction</div>', unsafe_allow_html=True)
st.divider()

# ------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("📋 Applicant Information")

age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Annual Income", 0, 1000000, 50000)
loan_amount = st.sidebar.number_input("Loan Amount", 0, 1000000, 20000)
credit_score = st.sidebar.number_input("Credit Score", 300, 850, 650)
interest_rate = st.sidebar.number_input("Interest Rate (%)", 0.0, 50.0, 10.0)
dti_ratio = st.sidebar.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
months_employed = st.sidebar.number_input("Months Employed", 0, 600, 24)
num_credit_lines = st.sidebar.number_input("Number of Credit Lines", 0, 20, 3)
loan_term = st.sidebar.number_input("Loan Term (Months)", 1, 480, 36)

education = st.sidebar.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
employment_type = st.sidebar.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
has_mortgage = st.sidebar.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Home", "Auto", "Business", "Education"])
has_cosigner = st.sidebar.selectbox("Has Co-Signer", ["Yes", "No"])

predict_button = st.sidebar.button("🚀 Analyze Risk")

# ------------------------------------------------
# Main Dashboard
# ------------------------------------------------
col1, col2 = st.columns(2)

if predict_button:

    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [num_credit_lines],
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [dti_ratio],
        "Education": [education],
        "EmploymentType": [employment_type],
        "MaritalStatus": [marital_status],
        "HasMortgage": [has_mortgage],
        "HasDependents": [has_dependents],
        "LoanPurpose": [loan_purpose],
        "HasCoSigner": [has_cosigner]
    })

    probability = float(model.predict_proba(input_data)[0][1])

    # ------------------------------------------------
    # Gauge Chart
    # ------------------------------------------------
    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 60], 'color': "orange"},
                    {'range': [60, 100], 'color': "red"}
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------
    # Risk Classification Card
    # ------------------------------------------------
    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Risk Assessment")

        if probability > 0.6:
            st.error("🔴 High Risk Applicant")
        elif probability > 0.35:
            st.warning("🟠 Moderate Risk Applicant")
        else:
            st.success("🟢 Low Risk Applicant")

        st.metric("Default Probability", f"{probability:.2%}")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Enter details in sidebar and click Analyze Risk.")

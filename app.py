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
# Load Preprocessor & Model
# ------------------------------------------------
preprocessor = joblib.load("preprocessor.pkl")

model = xgb.XGBClassifier()
model.load_model("model.json")

# ------------------------------------------------
# Business Rule Engine
# ------------------------------------------------
def apply_business_rules(age, employment_type, dti_ratio,
                         loan_term, loan_purpose,
                         has_cosigner, credit_score,
                         income, loan_amount):

    # -----------------------------------------
    # Global Age Policy
    # -----------------------------------------
    retirement_age = 60
    age_at_maturity = age + (loan_term / 12)

    if age < 21:
        return "REJECT", "Applicant below minimum legal age"

    if age_at_maturity > retirement_age:
        return "REVIEW", "Loan extends beyond retirement age"

    if age > 55:
        return "REVIEW", "Applicant close to retirement age"

    # -----------------------------------------
    # Credit Score Bands (Realistic)
    # -----------------------------------------
    if credit_score < 550:
        return "REJECT", "Very poor credit history"

    if credit_score < 600:
        return "HIGH_RISK", "Subprime credit score"

    # -----------------------------------------
    # Loan-to-Income Ratio (Important Metric)
    # -----------------------------------------
    if income > 0:
        lti_ratio = loan_amount / income
        if lti_ratio > 5:
            return "REJECT", "Loan amount too high compared to income"
        if lti_ratio > 3:
            return "HIGH_RISK", "High Loan-to-Income ratio"

    # -----------------------------------------
    # DTI (Debt-to-Income) Bands
    # -----------------------------------------
    if dti_ratio > 0.65:
        return "REJECT", "Excessive Debt-to-Income ratio"

    if dti_ratio > 0.5:
        return "HIGH_RISK", "High Debt burden"

    # -----------------------------------------
    # Employment Stability
    # -----------------------------------------
    if employment_type == "Unemployed":
        if loan_purpose in ["Auto", "Business"]:
            return "REJECT", "Stable income required"
        else:
            return "REVIEW", "No active income source"

    # -----------------------------------------
    # Loan Purpose Specific Policies
    # -----------------------------------------

    # 🚗 AUTO LOAN
    if loan_purpose == "Auto":
        if loan_term > 84:
            return "REVIEW", "Auto tenure above standard limit"
        if credit_score < 620:
            return "HIGH_RISK", "Auto loan with weak credit profile"

    # 🏠 HOME LOAN
    if loan_purpose == "Home":
        if loan_term > 360:
            return "REVIEW", "Home tenure unusually long"
        if dti_ratio > 0.45:
            return "HIGH_RISK", "High DTI for mortgage approval"

    # 💼 BUSINESS LOAN
    if loan_purpose == "Business":
        if credit_score < 650:
            return "HIGH_RISK", "Business loans require stronger credit"
        if lti_ratio > 4:
            return "REVIEW", "Aggressive business leverage"

    # 🎓 EDUCATION LOAN
    if loan_purpose == "Education":
        if has_cosigner == "No" and credit_score < 680:
            return "REJECT", "Co-signer required for this profile"
        if age > 40:
            return "REVIEW", "Education loan age atypical"

    return None, None

# ------------------------------------------------
# Header
# ------------------------------------------------
st.markdown('<div class="title">💳 AI Credit Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid ML + Rule-Based Loan Decision System</div>', unsafe_allow_html=True)
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
# Main Layout
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

    processed_data = preprocessor.transform(input_data)
    probability = float(model.predict_proba(processed_data)[0][1])

    rule_flag, rule_message = apply_business_rules(
    age,
    employment_type,
    dti_ratio,
    loan_term,
    loan_purpose,
    has_cosigner,
    credit_score,
    income,
    loan_amount
    )

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
    # Risk Card
    # ------------------------------------------------
    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Final Decision")

        if rule_flag == "REJECT":
            st.error("⛔ Loan Rejected (Policy Rule)")
            st.caption(rule_message)

        elif rule_flag == "HIGH_RISK":
            st.warning("⚠️ High Risk (Policy Override)")
            st.caption(rule_message)
            st.metric("ML Default Probability", f"{probability:.2%}")

        else:
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
import streamlit as st
import joblib
import pandas as pd
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
# Custom CSS (Enhanced)
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
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    font-size:18px;
    border-radius:12px;
    height:3em;
    width:100%;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load Model & Preprocessor
# ------------------------------------------------
preprocessor = joblib.load("preprocessor.pkl")

model = xgb.XGBClassifier()
model.load_model("model.json")

# ------------------------------------------------
# Business Rules
# ------------------------------------------------
def apply_business_rules(age, employment_type, dti_ratio,
                         loan_term, loan_purpose,
                         has_cosigner, credit_score,
                         income, loan_amount):

    if age < 21:
        return "REJECT", "Applicant below minimum legal age"

    if age > 55:
        return "REJECT", "Applicants above 55 years are not eligible for loans"

    if credit_score < 550:
        return "REJECT", "Very poor credit history"

    if dti_ratio > 0.65:
        return "REJECT", "Excessive Debt-to-Income ratio"

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
loan_term = st.sidebar.number_input("Loan Term (Months)", 1, 480, 36)

education = st.sidebar.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
employment_type = st.sidebar.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
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
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [dti_ratio],
        "Education": [education],
        "EmploymentType": [employment_type],
        "LoanPurpose": [loan_purpose],
        "HasCoSigner": [has_cosigner]
    })

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

    # ----------------------------------------
    # If Hard Rejection → Stop ML
    # ----------------------------------------
    if rule_flag == "REJECT":
        with col2:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.error("⛔ Loan Rejected (Policy Rule)")
            st.caption(rule_message)
            st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ----------------------------------------
    # ML Prediction
    # ----------------------------------------
    processed_data = preprocessor.transform(input_data)
    probability = float(model.predict_proba(processed_data)[0][1])

    # Dynamic Gauge Color
    if probability < 0.35:
        bar_color = "#00ff88"
        decision_label = "🟢 Low Risk Applicant"
    elif probability < 0.6:
        bar_color = "#ffb703"
        decision_label = "🟠 Moderate Risk Applicant"
    else:
        bar_color = "#ff4d4d"
        decision_label = "🔴 High Risk Applicant"

    # ----------------------------------------
    # Gauge
    # ----------------------------------------
    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [0, 30], 'color': "#00ff88"},
                    {'range': [30, 60], 'color': "#ffb703"},
                    {'range': [60, 100], 'color': "#ff4d4d"}
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------
    # Decision Card
    # ----------------------------------------
    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Final Decision")

        st.markdown(f"### {decision_label}")
        st.metric("Default Probability", f"{probability:.2%}")

        st.divider()
        st.subheader("📌 Key Metrics")
        st.write(f"• Credit Score: {credit_score}")
        st.write(f"• Debt-to-Income: {dti_ratio}")
        st.write(f"• Loan-to-Income: {round(loan_amount/income,2) if income>0 else 0}")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Enter details in sidebar and click Analyze Risk.")
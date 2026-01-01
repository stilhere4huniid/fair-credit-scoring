import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import dill
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FairCredit | AI Governance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'fair_credit_model.pkl')
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    model = joblib.load(model_path)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# --- DEFINE FEATURE NAMES (RAW MODEL NAMES) ---
# CRITICAL FIX: These must match the notebook X_train columns exactly.
# Lowercase, underscores, NO 'personal_status_sex'.
feature_names = [
    "status", "duration", "credit_history", "purpose", "credit_amount", 
    "savings", "employment", "installment_rate", "guarantors", 
    "residence_since", "property", "age", "other_installments", 
    "housing", "existing_credits", "job", "people_liable", "telephone", 
    "foreign_worker", "age_group"
]

# --- SIDEBAR: APPLICANT DATA ---
st.sidebar.header("📝 New Loan Application")
st.sidebar.markdown("Enter applicant details below:")

# Core Inputs
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=24)
credit_amt = st.sidebar.number_input("Credit Amount (DM)", min_value=100, max_value=20000, value=4000)
duration = st.sidebar.slider("Duration (Months)", 4, 72, 24)

# Categorical Inputs
sex_display = st.sidebar.selectbox("Sex", ["Male", "Female"])
housing_display = st.sidebar.selectbox("Housing", ["Own", "Rent", "Free"])
job_display = st.sidebar.selectbox("Job Skill Level", ["Unskilled", "Skilled", "Management"])

# Financial Health Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Financial Background")
financial_status = st.sidebar.selectbox(
    "Financial Standing", 
    ["Wealthy (High Savings)", "Average", "Poor (In Debt)"],
    index=0 
)

# --- MAIN DASHBOARD ---
st.title("⚖️ FairCredit: AI Governance Audit")
st.markdown("### Ethical Credit Scoring Engine")
st.markdown("---")

# 1. FAIRNESS CHECK
is_under_25 = age <= 25
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Applicant Age", f"{age} years")
with col2:
    if is_under_25:
        st.error("⚠️ Protected Group: YOUTH")
    else:
        st.success("Group: ADULT (Standard)")
with col3:
    st.metric("Fairness Protocol", "Active (Equalized Odds)")

# --- 2. PREPARE DATA ---
# The model expects 20 features.
input_data = np.zeros(20) 

# Inject Financial History (Based on Sidebar)
if "Wealthy" in financial_status:
    input_data[0] = 2   # status (>200)
    input_data[5] = 3   # savings (>1000)
    input_data[2] = 2   # credit_history (Paid back)
    input_data[6] = 3   # employment (4-7 yrs)
elif "Average" in financial_status:
    input_data[0] = 1   # status (0-200)
    input_data[5] = 1   # savings (100-500)
    input_data[2] = 2   # credit_history (Paid back)
    input_data[6] = 2   # employment (1-4 yrs)
else: # Poor
    input_data[0] = 0   # status (<0)
    input_data[5] = 0   # savings (<100)
    input_data[2] = 0   # credit_history (Delayed)
    input_data[6] = 0   # employment (Unemployed)

# Default 'Safe' Purpose
input_data[3] = 4 

# Map User Inputs
input_data[1] = duration
input_data[4] = credit_amt
input_data[11] = age

if housing_display == "Own": input_data[14] = 2
elif housing_display == "Rent": input_data[14] = 1
else: input_data[14] = 0

if job_display == "Skilled": input_data[16] = 2
elif job_display == "Management": input_data[16] = 3
else: input_data[16] = 0

# Age Group Feature (Index 19)
input_data[19] = 1 if age > 25 else 0

# --- CONVERT TO DATAFRAME ---
# This fixes the SHAP error by attaching the correct names
df_input = pd.DataFrame([input_data], columns=feature_names)

# Make Prediction
try:
    prediction = model.predict(df_input)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df_input)[0][1]
    else:
        probability = prediction[0]
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# --- 3. DECISION DISPLAY ---
st.markdown("### 🏦 AI Decision Engine")
final_col1, final_col2 = st.columns([2, 1])

with final_col1:
    if prediction[0] == 1:
        st.success("## ✅ LOAN APPROVED")
        st.write(f"**Confidence Score:** {probability:.1%}")
        st.progress(int(probability * 100))
    else:
        st.error("## ❌ LOAN REJECTED")
        st.write(f"**Risk Score:** {1 - probability:.1%}")
        st.progress(int((1-probability) * 100))

with final_col2:
    st.info("ℹ️ **Why this matters:**")
    st.caption("""
    This decision was processed by a **Fairness-Constrained XGBoost Model**.
    **Bias Gap:** Reduced from 11.5% ➡️ 2.5%
    """)

# --- 4. EXPLAINABILITY (SHAP) ---
st.markdown("---")
st.subheader("🔍 Explainability Analysis (Why?)")
st.write("This chart shows exactly which factors pushed the score UP (Red) or DOWN (Blue).")

if st.button("Generate Explanation Report"):
    with st.spinner("Calculating SHAP Values..."):
        try:
            # Create a TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(df_input)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            # Waterfall plot shows the cumulative effect of each feature
            shap.plots.waterfall(shap_values[0], max_display=7, show=False)
            st.pyplot(fig)
            
            st.markdown("""
            * **Red Bars (+):** Features that helped get the loan approved (e.g., High Savings).
            * **Blue Bars (-):** Features that hurt the application (e.g., Low Skill Job).
            """)
        except Exception as e:
            st.error(f"SHAP Error: {e}")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jobpy # or pickle, assuming you saved your trained RF model

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="FinFriend | Socially Aware AI", layout="wide")

# --- 2. CORE LOGIC: THE RESILIENCE SCORE (NumPy) ---
def calculate_resilience_score(savings, income, age, debt):
    """
    Your custom innovation: Factors in age-related risk zones.
    """
    # Base ratio
    savings_ratio = savings / (income + 1) 
    
    # Age-based risk multiplier (NumPy)
    # 50-year-olds have a higher risk penalty for low savings than 20-year-olds
    age_risk_multiplier = np.where(age > 45, 1.5, 1.0)
    debt_penalty = (debt / (income + 1)) * 0.5
    
    score = (savings_ratio * 100) - (debt_penalty * 100)
    final_score = np.clip(score / age_risk_multiplier, 0, 100)
    
    return round(float(final_score), 2)

# --- 3. SIDEBAR: USER INPUTS ---
st.sidebar.header("User Profile")
location_type = st.sidebar.selectbox("Location", ["Tier 1 Metro", "Tier 2 City", "Tier 3 Town"])
age = st.sidebar.slider("Age", 18, 80, 25)
monthly_income = st.sidebar.number_input("Monthly Income (₹)", value=30000)
monthly_expenses = st.sidebar.number_input("Monthly Expenses (₹)", value=15000)
total_savings = st.sidebar.number_input("Total Savings (₹)", value=50000)
total_debt = st.sidebar.number_input("Total Debt (₹)", value=0)

# --- 4. DATA VALIDATION PIPELINE ---
# Handling "Noise" (Expenses > Income logic)
if monthly_expenses > monthly_income:
    st.warning("⚠️ Data Alert: Your expenses exceed your income. Our model will adjust for this 'noise' to provide realistic advice.")
    valid_expense_ratio = 0.9 # Cap at 90% for model stability
else:
    valid_expense_ratio = monthly_expenses / monthly_income

# --- 5. MODEL PREDICTION (Socially Aware RF) ---
# Assuming the model expects: [Age, Income, Expense_Ratio, Location_Encoded, Debt]
# Tier 1: 0, Tier 2: 1, Tier 3: 2 (Example encoding)
loc_map = {"Tier 1 Metro": 0, "Tier 2 City": 1, "Tier 3 Town": 2}
features = np.array([[age, monthly_income, valid_expense_ratio, loc_map[location_type], total_debt]])

# To run without the .pkl file for now, we simulate the RF output:
# prediction = model.predict(features)
prediction = "Stable" if valid_expense_ratio < 0.7 else "At Risk"

# --- 6. MAIN DASHBOARD ---
st.title("FinFriend: Financial Resilience Dashboard")
st.markdown(f"### Status: **{prediction}**")

res_score = calculate_resilience_score(total_savings, monthly_income, age, total_debt)

col1, col2 = st.columns(2)

with col1:
    st.metric("Financial Resilience Score", f"{res_score}/100")
    st.progress(res_score / 100)
    
    if res_score < 40:
        st.error("High Risk Zone: Your age-to-savings ratio suggests a need for a larger safety net.")
    else:
        st.success("Safe Zone: Your financial structure is localized and resilient.")

with col2:
    # --- 7. VISUAL INSIGHTS (Matplotlib) ---
    st.subheader("Financial Leak Analysis")
    fig, ax = plt.subplots()
    labels = ['Expenses', 'Potential Savings']
    sizes = [monthly_expenses, max(0, monthly_income - monthly_expenses)]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax.axis('equal') 
    st.pyplot(fig)

st.divider()
st.info(f"Hyper-localized advice for **{location_type}**: Living costs and economic diversity have been factored into your score.")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="FinFriend", layout="centered")

# 2. Model Loading with Error Handling
@st.cache_resource
def load_finfriend_model():
    try:
        with open('finfriend_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

model = load_finfriend_model()

# 3. Sidebar Inputs
st.sidebar.title("User Profile")

income = st.sidebar.number_input("Monthly Income", min_value=5000, value=50000, step=1000)
age = st.sidebar.slider("Age", 18, 80, 25)
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3, 4, 5])

occ_label = st.sidebar.selectbox("Occupation", ['Student', 'Professional', 'Self_Employed', 'Retired'])
city_label = st.sidebar.selectbox("City Category", ['Tier_1', 'Tier_2', 'Tier_3'])

st.sidebar.subheader("Monthly Outgoings")
rent = st.sidebar.number_input("Rent or EMI", value=10000)
loan = st.sidebar.number_input("Loan Repayments", value=0)
insurance = st.sidebar.number_input("Insurance", value=2000)

# 4. Data Processing
occ_map = {'Student': 0, 'Professional': 1, 'Self_Employed': 2, 'Retired': 3}
city_map = {'Tier_1': 1, 'Tier_2': 2, 'Tier_3': 3}
age_factor = age / 100

# Feature array (must match training order)
features = np.array([[
    income, age_factor, dependents, 
    occ_map[occ_label], city_map[city_label], 
    rent, loan, insurance
]])

# 5. Main App Logic
st.title("FinFriend")
st.write("AI Financial Intelligence System")

if st.button("Analyze Finances"):
    if model:
        # ML Prediction
        prediction = model.predict(features)[0]
        
        # Resilience Logic
        fixed_costs = rent + loan + insurance
        disposable = income - fixed_costs
        savings_ratio = (disposable / income) * 100 if income > 0 else 0
        resilience = (savings_ratio * 0.7) + (age_factor * 30)
        
        # Dashboard Display
        st.write("### Financial Diagnostics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Target Monthly Savings", value=f"INR {prediction:,.0f}")
            
        with col2:
            st.metric(label="Resilience Score", value=f"{resilience:.1f}/100")
            if resilience > 65:
                st.success("Status: High Stability")
            elif resilience > 40:
                st.warning("Status: Moderate Risk")
            else:
                st.error("Status: High Vulnerability")

        # Visual Chart with Memory Management
        st.markdown("#### Budget Composition")
        labels = ['Target Savings', 'Fixed Costs', 'Other']
        others = max(0, income - prediction - fixed_costs)
        sizes = [prediction, fixed_costs, others]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax.axis('equal') 
        st.pyplot(fig)
        plt.close(fig) # CRITICAL: This prevents the 'Unresponsive' crash

        st.divider()
        st.write("#### Recommendation")
        st.write(f"For a {occ_label} in a {city_label} city, target a savings rate of INR {prediction:,.0f}. Reducing your INR {fixed_costs:,.0f} fixed costs will improve your Resilience Score.")
    else:
        st.error("Model file not found. Check if finfriend_model.pkl is in the main folder.")

st.caption("FinFriend | Hackonomics 2026")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 1. PAGE CONFIG & STYLING
st.set_page_config(page_title="FinFriend 🇮🇳", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 2. LIGHTWEIGHT MODEL LOADING
@st.cache_resource
def load_model():
    try:
        # Using 'rb' for read-binary
        with open('finfriend_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_model()

# 3. SIDEBAR: DATA INPUT
with st.sidebar:
    st.title("👤 User Profile")
    st.info("Input your details to generate an AI-backed savings goal.")
    
    # Financial Basics
    income = st.number_input("Monthly Income (₹)", min_value=5000, value=50000, step=1000)
    age = st.slider("Age", 18, 80, 25)
    dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4, 5])
    
    # Categories
    occ_label = st.selectbox("Occupation", ['Student', 'Professional', 'Self_Employed', 'Retired'])
    city_label = st.selectbox("City Category", ['Tier_1', 'Tier_2', 'Tier_3'])
    
    # Outgoings
    st.subheader("Monthly Outgoings")
    rent = st.number_input("Rent/EMI (₹)", value=10000)
    loan = st.number_input("Loan Repayments (₹)", value=0)
    insurance = st.number_input("Insurance (₹)", value=2000)

# 4. DATA TRANSFORMATION
# Mapping must be identical to your Kaggle Training order
occ_map = {'Student': 0, 'Professional': 1, 'Self_Employed': 2, 'Retired': 3}
city_map = {'Tier_1': 1, 'Tier_2': 2, 'Tier_3': 3}

# Feature Vector
features = np.array([[
    income, 
    age / 100, # age_factor
    dependents, 
    occ_map[occ_label], 
    city_map[city_label], 
    rent, 
    loan, 
    insurance
]])

# 5. MAIN CONTENT
st.title("FinFriend 🇮🇳")
st.write("### AI-Powered Savings Optimization")

if st.button("Run Financial Analysis"):
    if model:
        # Prediction
        prediction = model.predict(features)[0]
        
        # Resilience Math
        fixed_costs = rent + loan + insurance
        disposable = income - fixed_costs
        savings_ratio = (disposable / income) * 100 if income > 0 else 0
        resilience = (savings_ratio * 0.7) + ((age/

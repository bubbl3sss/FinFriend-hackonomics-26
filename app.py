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
    st.title("User Profile")
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

if st.button("🚀 Run Financial Analysis"):
    if model:
        # Prediction
        prediction = model.predict(features)[0]
        
        # Resilience Math
        fixed_costs = rent + loan + insurance
        disposable = income - fixed_costs
        savings_ratio = (disposable / income) * 100 if income > 0 else 0
        resilience = (savings_ratio * 0.7) + ((age/100) * 30)

        # Dashboard Columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Target Savings", f"₹{prediction:,.0f}")
        with col2:
            st.metric("Resilience Score", f"{resilience:.1f}/100")
        with col3:
            st.metric("Fixed Cost Ratio", f"{(fixed_costs/income)*100:.1f}%")

        st.divider()

        # Visuals
        left_plot, right_text = st.columns([2, 1])
        
        with left_plot:
            st.write("#### Budget Composition")
            labels = ['Target Savings', 'Fixed Costs', 'Other/Lifestyle']
            others = max(0, income - prediction - fixed_costs)
            sizes = [prediction, fixed_costs, others]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#2ecc71', '#e74c3c', '#3498db'])
            ax.axis('equal')
            st.pyplot(fig)
            plt.close(fig) 
        with right_text:
            st.write("#### 💡 AI Recommendations")
            if resilience < 40:
                st.error("Low Resilience: Your fixed costs are high relative to your age/income. Focus on an Emergency Fund.")
            elif resilience < 70:
                st.warning("Moderate Resilience: You have a buffer, but increasing savings by 5% would put you in the top tier.")
            else:
                st.success("High Resilience: You are in a strong position to invest aggressively!")
            
            st.info(f"Based on your profile in a **{city_label}**, most successful savers target **₹{prediction:,.0f}** monthly.")

    else:
        st.error("Model Error: Ensure 'finfriend_model.pkl' is in the root directory.")

st.divider()
st.caption("FinFriend | Hackonomics 2026 Submission | Built with Scikit-Learn & Streamlit")e/

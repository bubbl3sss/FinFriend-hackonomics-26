import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_finfriend_model():
    try:
        with open('finfriend_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_finfriend_model()


st.sidebar.title("User Profile")
st.sidebar.info("Adjust these to see your personalized savings target.")

income = st.sidebar.number_input("Monthly Income (₹)", min_value=5000, value=50000, step=1000)
age = st.sidebar.slider("Age", 18, 80, 25)
dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])

occ_label = st.sidebar.selectbox("Occupation", ['Student', 'Professional', 'Self_Employed', 'Retired'])
city_label = st.sidebar.selectbox("City Category", ['Tier_1', 'Tier_2', 'Tier_3'])

st.sidebar.subheader("Monthly Outgoings")
rent = st.sidebar.number_input("Rent/EMI (₹)", value=10000)
loan = st.sidebar.number_input("Loan Repayments (₹)", value=0)
insurance = st.sidebar.number_input("Insurance (₹)", value=2000)


occ_map = {'Student': 0, 'Professional': 1, 'Self_Employed': 2, 'Retired': 3}
city_map = {'Tier_1': 1, 'Tier_2': 2, 'Tier_3': 3}
age_factor = age / 100

features = np.array([[
    income, age_factor, dependents, 
    occ_map[occ_label], city_map[city_label], 
    rent, loan, insurance
]])

st.title("FinFriend")
st.write("### AI-Powered Financial Intelligence")

if st.button("Analyze My Finances"):
    if model:
        prediction = model.predict(features)[0]
        
        disposable = income - (rent + loan + insurance)
        savings_ratio = (disposable / income) * 100 if income > 0 else 0
        resilience = (savings_ratio * 0.7) + (age_factor * 30)
        
        st.write("### Your Financial Diagnostics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Target Monthly Savings", value=f"₹{prediction:,.0f}")
            st.info(f"Goal: { (prediction/income)*100 :.1f}% of income.")
            
        with col2:
            st.metric(label="Resilience Score", value=f"{resilience:.1f}/100")
            if resilience > 65:
                st.success("Status: High Stability")
            elif resilience > 40:
                st.warning("Status: Moderate Risk")
            else:
                st.error("Status: High Vulnerability")

        st.markdown("Monthly Budget Composition")
        labels = ['Target Savings', 'Fixed Costs', 'Other/Lifestyle']
    
        others = max(0, income - prediction - (rent + loan + insurance))
        sizes = [prediction, (rent + loan + insurance), others]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax.axis('equal') 
        st.pyplot(fig)

        st.divider()
        st.write("#### Recommendation")
        st.write(f"As a **{occ_label}** in a **{city_label}** city, you should target a savings rate of **₹{prediction:,.0f}**. Reducing your **₹{rent+loan:,.0f}** fixed liability could significantly boost your Resilience Score.")
    else:
        st.error("Model failed to load. Please check your .pkl file.")

st.caption("FinFriend | Hackonomics 2026 Submission")

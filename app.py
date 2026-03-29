import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import urllib.request
import os

# 1. Page Config
st.set_page_config(page_title="FinFriend", layout="centered")

# 2. THE CLOUD LOADER (Bypasses GitHub Size Limits)
@st.cache_resource
def load_model_from_url():
    model_path = 'finfriend_model.pkl'
    
    # If the model isn't in the repo, download it from your direct link
    if not os.path.exists(model_path):
        # REPLACE THE URL BELOW with your direct download link
        # For Google Drive, use: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
        url = 'https://drive.google.com/file/d/1Zp_XWjNj6gH6KJpvhr8mMvjl-kTHc6EF/view?usp=drive_link'
        try:
            with st.spinner('Loading AI Brain from Cloud...'):
                urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"Download Failed: {e}")
            return None

    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

model = load_model_from_url()

# 3. Sidebar Inputs
st.sidebar.title("User Profile")
income = st.sidebar.number_input("Monthly Income", min_value=5000, value=50000)
age = st.sidebar.slider("Age", 18, 80, 25)
occ_label = st.sidebar.selectbox("Occupation", ['Student', 'Professional', 'Self_Employed', 'Retired'])
city_label = st.sidebar.selectbox("City Category", ['Tier_1', 'Tier_2', 'Tier_3'])

st.sidebar.subheader("Outgoings")
rent = st.sidebar.number_input("Rent", value=10000)
loan = st.sidebar.number_input("Loan", value=0)
insurance = st.sidebar.number_input("Insurance", value=2000)

# 4. Processing
occ_map = {'Student': 0, 'Professional': 1, 'Self_Employed': 2, 'Retired': 3}
city_map = {'Tier_1': 1, 'Tier_2': 2, 'Tier_3': 3}

features = np.array([[
    income, age/100, 0, # age_factor and placeholder for dependents
    occ_map[occ_label], city_map[city_label], 
    rent, loan, insurance
]])

# 5. UI Logic
st.title("FinFriend")

if st.button("Analyze Finances"):
    if model:
        prediction = model.predict(features)[0]
        st.metric("Target Monthly Savings", f"INR {prediction:,.0f}")
        
        # Simple Chart
        fig, ax = plt.subplots()
        ax.pie([prediction, (rent+loan+insurance)], labels=['Savings', 'Costs'])
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Please check your Model URL in the code.")

st.caption("FinFriend | Hackonomics 2026")

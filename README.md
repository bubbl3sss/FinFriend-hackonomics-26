# FinFriend
FiFinFriend is a specialized Machine Learning application designed for the Indian economic landscape. While most apps simply track where your money went, FinFriend uses AI to tell you where your money should go.

By leveraging a dataset of 11,000+ Indian financial profiles, it provides hyper-localized savings targets and a custom "Financial Resilience Score" to help users navigate the unique costs of Tier 1, 2, and 3 cities.

## Features

1. The "Ideal Savings" Oracle
Using a Random Forest Regressor, FinFriend predicts a user's "Ideal Monthly Savings" based on their income, age, and occupation. It benchmarks you against the top 20% of savers in your specific demographic, giving you a goal that is both ambitious and achievable.

2. Financial Resilience Scoring (The NumPy Engine)
Unlike a standard credit score, our Resilience Score (0-100) is a custom metric engineered using NumPy. It calculates the ratio of disposable income to total earnings, weighted against age-based stability factors. It tells you how "weather-proof" your finances are against emergencies.

3. Regional Optimization (Bharat-Centric)
Living in Mumbai (Tier 1) is not the same as living in Nagpur (Tier 2). FinFriend adjusts its logic based on the City Tier, ensuring that rent and lifestyle expectations are realistic for the user's location.

## Tech Stack
Data Engineering:
-Numpy for feature scaling
-Max min normalizations and transformations

Machine Learning Pipeline
-Random Forest Regressor 
-100 Decision trees to minimize variance and handle non-linear financial relationships.
-Streamlit for deployment

## LINK TO KAGGLE NOTEBOOK: https://www.kaggle.com/code/bubbl3ss/finfriend-hackonomics-26


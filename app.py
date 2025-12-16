import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

#Load the trained model
model = load_model('models/ann_model.h5')
# Load the scaler and encoders
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('models/onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

### streamlit app
# st.title("Customer Churn Prediction")

# #User input
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender=st.selectbox("Gender", ["Male", "Female"])
age=st.slider("Age", min_value=18, max_value=100, value=30)
##tenure=st.number_input("Tenure", min_value=0, max_value=10, value=5)
balance=st.number_input("Balance", min_value=0.0, value=1000.00  )  
credit_score=st.number_input("Credit Score", min_value=300, max_value=850, value=600)
estimated_salary=st.number_input("Estimated Salary", min_value=0.0, value=50000.00)
tenure=st.slider("Tenure", min_value=0, max_value=10, value=5)
num_of_products=st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card=st.selectbox("Has Credit Card", [0, 1]) 
is_active_member=st.selectbox("Is Active Member", [0, 1])

gender=label_encoder.transform([gender])[0]
geography_encoded = onehot_encoder.transform([[geography]]).toarray()[0]
geography_df = pd.DataFrame([geography_encoded], columns=onehot_encoder.get_feature_names_out(['Geography']))
if st.button("Predict Churn"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)

# Scale the input data
    input_data_scaled=scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]*100
    st.write(f"churn probability: {churn_probability:.4f}%")
    if churn_probability > 50:
           st.write(f"The customer is likely to churn ")
    else:
            st.write(f"The customer is unlikely to churn")
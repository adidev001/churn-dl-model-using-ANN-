import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('lable_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('ohe_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')

# User inputs
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Add prediction button and logic
if st.button('Predict Churn'):
    # Prepare input data
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }

    # Create DataFrame and preprocess
    input_df = pd.DataFrame([input_data])
    input_df['Gender'] = label_encoder_gender.transform([input_data['Gender']])
    
    # Encode geography
    geo_encoded = ohe_geo.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, 
        columns=ohe_geo.get_feature_names_out(['Geography'])
    )
    
    # Combine and scale features
    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    churn_probability = prediction[0][0]
    
    # Display results
    st.header('Prediction Results')
    st.write(f'Churn Probability: {churn_probability:.2%}')
    
    if churn_probability > 0.5:
        st.error(' High Risk of Churn!')
    else:
        st.success(' Low Risk of Churn!')

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model_1.h5')

# Load the encoders and scaler
with open('scaler_1.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('one_hot_encoder_geography_1.pkl', 'rb') as f:
    one_hot_encoder_geography = pickle.load(f)

with open('label_encoder_gender_1.pkl', 'rb') as f:
    gender_label_encoder = pickle.load(f)

# ---------------- STREAMLIT UI ---------------- #

st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', gender_label_encoder.classes_)
age = st.slider('Age', 18, 70)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=600)
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# ---------------- DATA PROCESSING ---------------- #

# Encode Gender
gender_encoded = gender_label_encoder.transform([gender])[0]

# Create base input dataframe
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Gender': [gender_encoded],
    'Balance': [balance],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# One-hot encode Geography
geo_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()

geo_columns = one_hot_encoder_geography.get_feature_names_out(['Geography'])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

# Combine all features
input_df = pd.concat([input_df, geo_encoded_df], axis=1)

# Ensure correct column order (VERY IMPORTANT)
input_df = input_df[scaler.feature_names_in_]

# Scale the data
input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION ---------------- #

prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.write(f"Churn Probability: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.error(f"The customer is likely to churn ❌ (Probability: {prediction_prob:.2f})")
else:
    st.success(f"The customer is not likely to churn ✅ (Probability: {prediction_prob:.2f})")
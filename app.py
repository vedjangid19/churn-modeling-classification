import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np


# load model 
model = tf.keras.models.load_model('model.h5')

# load scaler, one_hot_encoder_geo and label_encoder_gender
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_coder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

# Streamlit app layout
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Input Data
# Example input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# condata encoded geo and inpur df
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
print("----------------------------------------------",input_data)

# Standard Scaler the input data
input_data_scaled = scaler.transform(input_data)

# prdiction
pred = model.predict(input_data_scaled)[0][0]

print(f"prediction: {pred}")
st.write(f"prediction score: {pred}")
if pred < 0.5:
    st.write("The customer is likely to Churn")
else:
    st.write("The customer is not likely to Churn")
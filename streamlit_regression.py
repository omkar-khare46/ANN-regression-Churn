import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

##Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

## load encoder and scaler files
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender =  pickle.load(file)
with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography =  pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler =  pickle.load(file)

## streamlit app
st.title('Estimated Salary Prediction')

###user input
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0]) 
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox  ('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is active member', [0,1])
exited = st.selectbox('Exited', [0,1])

### prepare the input data
input_data = pd.DataFrame({'CreditScore':[credit_score],	'Geography':[geography],	'Gender':[label_encoder_gender.transform([gender])[0]], 	'Age':[age],	'Tenure':[tenure],
              	'Balance':[balance],	'NumOfProducts':[num_of_products],	'HasCrCard':[has_cr_card],	'IsActiveMember':[is_active_member],	'Exited': [exited]})

## one hot encoded geography
geography_encoded = onehot_encoder_geography.transform([[geography]])
geo_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out())

input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df],axis=1)
input_data.drop('Geography', axis =1, inplace =True)

input_scaled = scaler.transform(input_data)

## predict the estimated salary
prediction = model.predict(input_scaled)
predicted_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary : ${predicted_salary:.2f}')
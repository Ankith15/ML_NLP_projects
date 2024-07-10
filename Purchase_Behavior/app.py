import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('Purchase_Behavior/model.pkl', 'rb') as f:
    mod = pickle.load(f)

with open('Purchase_Behavior/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Purchase_Behavior/LE.pkl', 'rb') as f:
    le = pickle.load(f)

with open('Purchase_Behavior/ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

st.title('Understanding the purchase behavior')

Age = st.number_input('Age', min_value=0, max_value=100, value=25)
Salary = st.number_input('Salary', min_value=0, max_value=100000000, value=22000)
Product_ID = st.selectbox('Product ID', ['P01', 'P02', 'P03'])
Price = st.number_input('Price', min_value=1, max_value=10000, value=110)
Gender = st.selectbox('Gender', ['Male', 'Female'])

if st.button("Predict"):
    input_data = pd.DataFrame([[Age, Salary, Product_ID, Price, Gender]], 
                              columns=['Age', 'Salary', 'Product ID', 'Price', 'Gender'])
    
    input_data[['Age', 'Salary', 'Price']] = scaler.transform(input_data[['Age', 'Salary', 'Price']])
    
    gender_binary = 1 if Gender == "Male" else 0
    gender_encoded_df = pd.DataFrame([[gender_binary]], columns=['Gender_Male'])
    
    product_id_encoded = le.transform(input_data['Product ID'])
    product_id_encoded_df = pd.DataFrame(product_id_encoded, columns=['Product_ID'], index=input_data.index)
    
    input_data = input_data.drop(columns=['Gender', 'Product ID'])
    
    input_data = pd.concat([input_data, gender_encoded_df, product_id_encoded_df], axis=1)
    
    input_data = input_data[['Age', 'Salary', 'Price', 'Gender_Male', 'Product_ID']]
    
    prediction = mod.predict(input_data)
    
    if prediction[0] == 1:
        st.success('Customer will purchase the item')
    else:
        st.warning('Customer will not purchase the item')
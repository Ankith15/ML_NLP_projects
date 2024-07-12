import streamlit as st
import pandas as pd
import numpy as np
import pickle

class PurchaseBehaviorApp:
    def __init__(self, model_path, scaler_path, le_path, oh_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)
        with open(oh_path, 'rb') as f:
            self.oh = pickle.load(f)

    def predict_purchase(self, Age, Salary, Product_ID, Price, Gender):
        input_data = pd.DataFrame([[Age, Salary, Product_ID, Price, Gender]], columns=['Age', 'Salary', 'Product ID', 'Price', 'Gender'])
        input_data[['Age', 'Salary', 'Price']] = self.scaler.transform(input_data[['Age', 'Salary', 'Price']])
        
        gender_binary = 1 if Gender == "Male" else 0
        gender_encoded_df = pd.DataFrame([[gender_binary]], columns=['Gender_Male'])
        
        product_id_encoded = self.le.transform(input_data['Product ID'])
        product_id_encoded_df = pd.DataFrame(product_id_encoded, columns=['Product ID'], index=input_data.index)
        
        input_data = input_data.drop(columns=['Gender', 'Product ID'])
        input_data = pd.concat([input_data, gender_encoded_df, product_id_encoded_df], axis=1)
        
        input_data = input_data[['Age', 'Salary', 'Product ID', 'Price', 'Gender_Male']]
        
        prediction = self.model.predict(input_data)
        return prediction[0]

    def run(self):
        st.title('Understanding the Purchase Behavior')
        Age = st.number_input('Age', min_value=0, max_value=100, value=25)
        Salary = st.number_input('Salary', min_value=0, max_value=100000000, value=22000)
        Product_ID = st.selectbox('Product ID', ['P01', 'P02', 'P03'])
        Price = st.number_input('Price', min_value=1, max_value=10000, value=110)
        Gender = st.selectbox('Gender', ['Male', 'Female'])

        if st.button("Predict"):
            prediction = self.predict_purchase(Age, Salary, Product_ID, Price, Gender)
            if prediction == 1:
                st.success('Customer will purchase the item')
            else:
                st.warning('Customer will not purchase the item')

app = PurchaseBehaviorApp('Purchase_Behavior/model.pkl', 'Purchase_Behavior/scaler.pkl', 'Purchase_Behavior/LE.pkl', 'Purchase_Behavior/ohe.pkl')
app.run()

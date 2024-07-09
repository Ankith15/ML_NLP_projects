import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model.pkl','rb') as f:
    mod = pickle.load(f)

with open('LE.pkl','rb') as f:
    le = pickle.load(f)

with open ('ohe.pkl','rb') as f:
    ohe = pickle.load(f)

# buileding the streamlit app

st.title('Understanding the purchase behaviour')

Age = st.number_input('Age',min_value=0,max_value=100,value=25)
Salary = st.number_input('Salary',min_value=0,max_value=100000000,value=22000)
Product_ID = st.selectbox('Product ID',['P01','P02','P03'])
Price = st.number_input('Price',min_value=1,max_value=10000,value=110)
Gender = st.selectbox('Gender',['Male','Female'])


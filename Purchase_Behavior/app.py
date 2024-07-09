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
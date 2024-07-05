import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('Purchase_Behavior\purchase_history.csv')
print(df.head())

oh = OneHotEncoder(drop='first')
df['Gender']=oh.fit_transform(df[['Gender']])

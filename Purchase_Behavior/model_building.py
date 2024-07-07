import pandas as pd 
import numpy as np
from Preprocessing import new_df

df = pd.read_csv('Purchase_Behavior\Purchase_data.csv')
df.drop(columns=['Unnamed: 0','Customer ID'],inplace=True)
# print(df.head(5))

x = df.drop(columns=['Purchased'])
print(x.head())
y = df['Purchased']
print(y.head())
import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('Purchase_Behavior\purchase_history.csv')
# print(df.head())
df.drop(columns=['Customer ID'])
oh = OneHotEncoder(drop='first')
Encoded =oh.fit_transform(df[['Gender']])
encoded_gender_df = pd.DataFrame(Encoded.toarray(), columns=oh.get_feature_names_out(['Gender']))

df = df.drop('Gender', axis=1)
df = pd.concat([df, encoded_gender_df], axis=1)

le = LabelEncoder()
df['Product ID'] = le.fit_transform(df['Product ID'])

scaler = StandardScaler()

df[['Age','Salary','Price']] = scaler.fit_transform(df[['Age','Salary','Price']])
new_df = df
print(new_df.head())

with open('Purchase_Behavior\scaler.pkl','wb') as f:
    pickle.dump(scaler,f)

# new_df.to_csv('Purchase_data.csv')
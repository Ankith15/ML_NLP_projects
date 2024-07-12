import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

class Preprocessing:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.scaler = StandardScaler()
        self.oh = OneHotEncoder(drop='first')
        self.le = LabelEncoder()

    def drop_columns(self, columns):
        self.df.drop(columns=columns, inplace=True)

    def encode_gender(self):
        encoded = self.oh.fit_transform(self.df[['Gender']])
        encoded_gender_df = pd.DataFrame(encoded.toarray(), columns=self.oh.get_feature_names_out(['Gender']))
        self.df = self.df.drop('Gender', axis=1)
        self.df = pd.concat([self.df, encoded_gender_df], axis=1)

    def encode_product_id(self):
        self.df['Product ID'] = self.le.fit_transform(self.df['Product ID'])

    def scale_features(self):
        self.df[['Age', 'Salary', 'Price']] = self.scaler.fit_transform(self.df[['Age', 'Salary', 'Price']])
    
    def save_encoders(self, scaler_path, oh_path, le_path):
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(oh_path, 'wb') as f:
            pickle.dump(self.oh, f)
        with open(le_path, 'wb') as f:
            pickle.dump(self.le, f)

    def save_preprocessed_data(self, filepath):
        self.df.to_csv(filepath, index=False)
    
    def preprocess(self):
        self.drop_columns(['Customer ID'])
        self.encode_gender()
        self.encode_product_id()
        self.scale_features()
        return self.df

preprocessor = Preprocessing('Purchase_Behavior/purchase_history.csv')
new_df = preprocessor.preprocess()
print(new_df.head())
preprocessor.save_encoders('Purchase_Behavior/scaler.pkl', 'Purchase_Behavior/ohe.pkl', 'Purchase_Behavior/LE.pkl')
preprocessor.save_preprocessed_data('Purchase_Behavior/Purchase_data.csv')

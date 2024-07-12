import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

class ModelBuilding:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.model = LogisticRegression()
        
    def preprocess_data(self):
        self.df.drop(columns=['Unnamed: 0', 'Customer ID'], inplace=True)
        self.X = self.df.drop(columns=['Purchased'])
        self.y = self.df['Purchased']

    def train_test_split(self, test_size=0.2, random_state=78):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
    
    def build_model(self):
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        cross_val_accuracy = np.mean(cross_val_score(self.model, self.X, self.y, cv=5))
        return accuracy, cross_val_accuracy

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

model_builder = ModelBuilding('Purchase_Behavior/Purchase_data.csv')
model_builder.preprocess_data()
model_builder.train_test_split()
model_builder.build_model()
accuracy, cross_val_accuracy = model_builder.evaluate_model()
print(f"Accuracy: {accuracy}, Cross-Validation Accuracy: {cross_val_accuracy}")
model_builder.save_model('Purchase_Behavior/model.pkl')

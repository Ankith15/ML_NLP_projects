import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

df = pd.read_csv('Purchase_Behavior\Purchase_data.csv')
df.drop(columns=['Unnamed: 0','Customer ID'],inplace=True)
# print(df.head(5))

x = df.drop(columns=['Purchased'])
print(x.head())
y = df['Purchased']
# print(y.head())

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=78)
print(x_test.shape)


# Model building

lr =  LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print (accuracy_score(y_test,y_pred))
print(np.mean(cross_val_score(lr,x,y,cv=5)))


with open('Purchase_Behavior\model.pkl','wb') as f:
    pickle.dump(lr,f)
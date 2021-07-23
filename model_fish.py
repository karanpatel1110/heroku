import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
data = pd.read_csv('Fish.csv')
data_head = data.head()
print(data_head)
x = data.drop(['Species'], axis = 1)
print(x)
#y = data.Species
#print(y)

le = LabelEncoder()
y = le.fit_transform(data.Species)
print(data.Species)

x_train , x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 1)
print(x_train)
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(acc)

pickle.dump(classifier,open("model_fish.pkl","wb"))
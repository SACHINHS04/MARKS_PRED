# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:14:15 2021

@author: sachin h s
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("./student_info.csv")
#print(dataset.head())

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

classifier=LinearRegression()
classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)


import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


    



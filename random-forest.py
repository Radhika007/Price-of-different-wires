#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:22:35 2019

@author: radhika
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#importing the dataset
dataset = pd.read_csv('wireDeets2.csv')
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values

#Categorical encoding
labelencoder = LabelEncoder()
X[:,2] = labelencoder.fit_transform(X[:,2])
Y = pd.DataFrame(X)
onehotencoder = OneHotEncoder(categorical_features=[2])
X = onehotencoder.fit_transform(X).toarray()

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

#Fitting the data
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#Accuracy score through r-square method
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

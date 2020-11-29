# Regresión Lineal Simple

'''
Este modelo predice el salario del empleado basado en la experiencia 
usando un modelo sencillo de regresión lineal.
'''

# Importando las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json

# Importando el dataset
dataset = pd.read_csv('datos.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Separando el dataset en conjunto de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Ajustando Regresión Lineal al conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediciendo resultados del set de prueba
y_pred = regressor.predict(X_test)

# Guardando el modelo usando pickle
pickle.dump(regressor, open('model.pkl','wb'))

# Cargando el modelo para comparar resultados
model = pickle.load( open('model.pkl','rb'))
print(model.predict([[1.8]]))

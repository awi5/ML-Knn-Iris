# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:43:23 2022

@author: DELL
"""

import streamlit as st
import pandas as pd
import PIL as Image
import joblib

#Loading Our final trained Knn model
model =  open("Knn_Classify.pkl", "rb" )
knn_clf = joblib.load(model)

st.title("Iris flower species Classification App")
st.sidebar.title("Features")


#Loading images
setosa = Image.open('Setosa.jpg')
versicolor= Image.open('Versicolor.jpg')
virginica = Image.open('Virginica.jpg')


#Initializing
parameter_list = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
parameter_input_values = []
parameter_default_values=['5.2', '3.2', '4.2', '1.2']
values=[]

#Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
    values =st.sidebar.slider(label=parameter,
                              key=parameter,
                              value=float(parameter_df),
                              min_value=0.0,
                              max_value=8.0,
                              step=-0.1)
    parameter_input_values.append(values)
    
input_variable = pd.DataFrame([parameter_input_values],
                              columns=parameter.list,
                              dtype=float)
st.write(input_variable)


if st.button("Click here to Classify"):
    prediction = knn_clf.predict(input_variable)
    
if prediction== 0:
    st.image(setosa)
elif prediction== 1:
    st.image(versicolor)  
else:
    st.image(virginica)
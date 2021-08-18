# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 19:18:17 2021

@author: sachin h s
"""

import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_marks(study_hours):
    
    """Let's predict marks 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: study_hours
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[study_hours]])
    print(prediction)
    return prediction



def main():
    st.title("LET'S PREDICT YOUR MARKS")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">MARKS PREDICTION ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    study_hours = st.text_input("STUDY HOURS","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_marks(study_hours)
    st.success('the predicted marks {}'.format(result))
    if st.button("About"):
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
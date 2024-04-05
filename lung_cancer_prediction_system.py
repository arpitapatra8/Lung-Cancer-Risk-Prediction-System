# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:17:53 2024

@author: admin
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('prediction_model.sav', 'rb'))

#creating function for prediction
def prediction(user_input):
    
    user_input = np.asarray(user_input)
    user_input = user_input.reshape(1,-1)

    # Predict using Naive Bayes model
    prediction = loaded_model.predict(user_input)
    
    if (prediction[0] == 0):
        return 'LOW RISK'
    elif (prediction[0] == 1):
        return 'MEDIUM RISK'
    else:
        return 'HIGH RISK'

    
def main():
    
    #title
    st.title('Lung Cancer Risk Prediction System')
    
    #input data from user
    age = int(st.number_input("Age", step=1))
    gender = int(st.number_input("Gender", step=1))
    air_pollution = int(st.number_input("Air Pollution", step=1))
    alcohol_consumption = int(st.number_input("Alcohol Consumption", step=1))
    dust_allergy = int(st.number_input("Dust Allergy", step=1))
    occupational_hazard = int(st.number_input("Occupational Hazard", step=1))
    genetic_risk = int(st.number_input("Genetic Risk", step=1))
    chronic_lung_disease = int(st.number_input("Chronic Lung Disease", step=1))
    balanced_diet = int(st.number_input("Balanced Diet", step=1))
    obesity = int(st.number_input("Obesity", step=1))
    smoking = int(st.number_input("Smoking", step=1))
    passive_smoking = int(st.number_input("Passive Smoking", step=1))
    chest_pain = int(st.number_input("Chest Pain", step=1))
    coughing_of_blood = int(st.number_input("Coughing of Blood", step=1))
    fatigue = int(st.number_input("Fatigue", step=1))
    weight_loss = int(st.number_input("Weight Loss", step=1))
    shortness_of_breath = int(st.number_input("Shortness of Breath", step=1))
    wheezing = int(st.number_input("Wheezing", step=1))
    difficulty_swallowing = int(st.number_input("Difficulty Swallowing", step=1))
    frequent_cold = int(st.number_input("Frequent Cold", step=1))
    dry_cough = int(st.number_input("Dry Cough", step=1))
    snoring = int(st.number_input("Snoring", step=1))

    
    # Code for prediction
    level = ''

    # Result
    if st.button('Prediction Result'):
        level = prediction([age, gender, air_pollution, alcohol_consumption, dust_allergy, occupational_hazard, genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoking, chest_pain, coughing_of_blood, fatigue, weight_loss, shortness_of_breath, wheezing, difficulty_swallowing, frequent_cold, dry_cough, snoring])

    # Display the success message with HTML formatting and the custom class
    st.markdown(f'<p class="custom-success">{level}</p>', unsafe_allow_html=True)
    
    
if __name__ == '__main__':
    main()

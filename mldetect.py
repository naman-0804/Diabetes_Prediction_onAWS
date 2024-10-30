import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Import joblib for loading the model

# Load your trained model
model = joblib.load('modelaws.joblib')  # Ensure the path to your joblib model is correct

# Streamlit application
st.title('Diabetes Detection App')

# Input fields for features
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=6)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=148)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=72)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=35)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=33.6)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.627)
age = st.number_input('Age', min_value=0, max_value=120, value=50)

# Prepare the input for prediction
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
input_data = pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Make predictions
if st.button('Predict'):
    prediction = model.predict(input_data)  # Use the loaded joblib model for prediction
    predicted_label = np.round(prediction[0]).astype(int)  # Assuming binary classification (0 or 1)

    # Display the prediction result
    st.write(f'Predicted Label: {predicted_label}')

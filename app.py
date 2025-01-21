import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_filename = 'model_uas.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
st.title('Insurance Charges Prediction')
st.write("Developed by Anindya Tara Danendra Alamsyah")
st.write("NIM: 2021230021")

# Input fields
age = st.number_input('Age', min_value=0, max_value=100, step=1)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
children = st.number_input('Number of Children', min_value=0, max_value=10, step=1)
smoker = st.selectbox('Smoker', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Submit button
if st.button('Predict Charges'):
    # Prepare the input data
    input_data = np.array([[age, sex, bmi, children, smoker]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f'The predicted insurance charge is ${prediction[0]:.2f}')

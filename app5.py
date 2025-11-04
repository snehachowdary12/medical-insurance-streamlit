import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.markdown("<h1 style='text-align: center; color: navy;'>Medical Insurance Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar input fields
st.sidebar.header("Enter Patient Information")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
bp = st.sidebar.selectbox("Blood Pressure Problems", [0, 1])
transplants = st.sidebar.selectbox("Any Transplants", [0, 1])
chronic = st.sidebar.selectbox("Any Chronic Diseases", [0, 1])
height = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
allergies = st.sidebar.selectbox("Known Allergies", [0, 1])
cancer = st.sidebar.selectbox("History of Cancer in Family", [0, 1])
surgeries = st.sidebar.number_input("Number of Major Surgeries", min_value=0, max_value=10, value=0)

# Prediction
if st.sidebar.button("Predict Premium"):
    input_data = np.array([[age, bp, transplants, chronic, height, weight, allergies, cancer, surgeries]])
    prediction = model.predict(input_data)[0]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: #e6f2ff; border-radius: 10px;'>
        <h2 style='color: green;'>Predicted Insurance Premium:</h2>
        <h1 style='color: darkblue;'>₹ {prediction:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)



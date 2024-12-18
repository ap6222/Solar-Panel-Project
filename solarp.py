import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained Gradient Boosting model
with open('gbr_model.pkl', 'rb') as model_file:
    gbr_model = pickle.load(model_file)

# Load the StandardScaler model
with open('scaler_model.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI for user input
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://plus.unsplash.com/premium_photo-1679917152411-353fd633e218?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mjl8fHNvbGFyJTIwZW5lcmd5fGVufDB8fDB8fHww');
        background-size: cover;
        background-position: center;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 3em;
        color: white;
        text-align: center;
        margin-top: 50px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .stButton>button {
        background-color: #ff8c00;
        color: white;
        font-size: 1.2em;
        border-radius: 10px;
        padding: 10px 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e07b00;
        cursor: pointer;
    }
    .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 5px;
        padding: 5px;
    }
    .stSelectbox select {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 5px;
        padding: 5px;
    }
    .stLabel {
        color: red;  /* Make input labels red */
    }
    .prediction {
        color: red;  /* Make prediction text red */
        text-align: center;  /* Center align prediction */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title section with a solar-related logo
st.markdown(
    """
    <div class="title">
        <img src="https://images.unsplash.com/photo-1466152187244-1cb1ac3dec04?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjB8fHVuc3BsYXNoJTIwbG9nbyUyMHNvbGFyJTIwcG93ZXIlMjBwcmVkaWN0aW9ufGVufDB8fDB8fHww" 
             width="50px" height="50px" style="vertical-align: middle; margin-right: 15px;">  
        Power Generation Prediction
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar details
st.sidebar.header("Project Details")
st.sidebar.write("**Solar Power Generation Prediction App**")
st.sidebar.write("This app uses a machine learning model to predict solar power generation based on various weather and environmental factors.")
st.sidebar.write("**Data Pipeline:**")
st.sidebar.write("1. Data Collection")
st.sidebar.write("2. EDA: Cleaning, Validation, Handling Missing Values, Outlier Treatment, Visualization")
st.sidebar.write("3. Data Transformation: Scaling, Encoding")
st.sidebar.write("4. Model Building and Evaluation")
st.sidebar.write("5. Hyperparameter Tuning")
st.sidebar.write("6. Deployment")
st.sidebar.write("**Model:** Gradient Boosting Regressor")
st.sidebar.write("**Accuracy:** Train R²: 0.9452, Test R²: 0.9163")
st.sidebar.write("**Created by:")
st.sidebar.write(" 1. Darshana Mahajan") 
st.sidebar.write(" 2. Atharva Pawar") 
st.sidebar.write(" 3. Akash RT") 
st.sidebar.write(" 4. Gurram Saisuchitra") 
st.sidebar.write(" 5. Paladugu.Venkata Bhanu Priyanka") 
st.sidebar.write(" 6. Nakirikanti Naga Tejaswini") 



# Mapping for 'sky_cover' from 0 to 4 (ordinal)
sky_cover_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

# Take user inputs
input_data = {
    'distance_to_solar_noon': st.number_input('Distance to Solar Noon (radians)', min_value=0.0, value=0.0, step=0.01),
    'temperature': st.number_input('Temperature (°C)', min_value=-50.0, value=25.0, step=0.1),
    'wind_direction': st.number_input('Wind Direction (°)', min_value=0, value=0, step=1),
    'wind_speed': st.number_input('Wind Speed (m/s)', min_value=0.0, value=10.0, step=0.1),
    'sky_cover': st.selectbox('Sky Cover', options=[0, 1, 2, 3, 4]),
    'humidity': st.number_input('Humidity (%)', min_value=0, value=75, step=1),
    'average_wind_speed_(period)': st.number_input('Average Wind Speed (m/s)', min_value=0.0, value=10.0, step=0.1),
    'average_pressure_(period)': st.number_input('Average Pressure (hPa)', min_value=0.0, value=1050.0, step=0.1)
}

# Map 'sky_cover' to 'sky_cover_encoded'
input_data['sky_cover_encoded'] = sky_cover_mapping[input_data['sky_cover']]
del input_data['sky_cover']  # Drop original 'sky_cover'

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure input columns match model training
expected_columns = ['distance_to_solar_noon', 'temperature', 'wind_direction', 'wind_speed', 'humidity',
                    'average_wind_speed_(period)', 'average_pressure_(period)', 'sky_cover_encoded']
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Prediction on button click
if st.button('Predict'):
    prediction = gbr_model.predict(input_scaled)
    st.markdown(
        f"""
        <div style="font-size: 1.5em; color: green; font-weight: bold; text-align: center; margin-top: 20px;">
            Predicted Power Generated: {prediction[0]:.2f} Joules
        </div>
        """,
        unsafe_allow_html=True
    )

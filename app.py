import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(page_title="Grade Impact Predictor", page_icon="🎓", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; background-color: #2e7d32; color: white; border-radius: 8px; }
    .result-card { padding: 30px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Title and Description
st.title("🎓 AI Usage & Academic Performance Predictor")
st.write("Predict the impact of AI tools on student grades using machine learning.")

# Sidebar Information
st.sidebar.header("About Model")
st.sidebar.info("Model Type: K-Neighbors Classifier")
st.sidebar.write("Target Classes: High, Medium, Low")

# UI Layout for Inputs
with st.container():
    st.subheader("📝 Student Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 15, 60, 20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        city = st.selectbox("City", ["New York", "London", "Mumbai", "Other"])
    
    with col2:
        edu = st.selectbox("Education", ["High School", "Undergraduate", "Postgraduate", "PhD"])
        tool = st.selectbox("AI Tool", ["ChatGPT", "Gemini", "Claude", "Other"])
    
    with col3:
        usage = st.slider("Daily Usage (Hours)", 0.0, 15.0, 2.0)
        purpose = st.selectbox("Primary Purpose", ["Research", "Coding", "Writing", "General"])

# Data Mapping (Required for KNN Models)
# Note: These maps should match the ones used during your model training
mapping = {
    "Male": 0, "Female": 1, "Other": 2,
    "High School": 0, "Undergraduate": 1, "Postgraduate": 2, "PhD": 3,
    "New York": 0, "London": 1, "Mumbai": 2, "Other": 3,
    "ChatGPT": 0, "Gemini": 1, "Claude": 2,
    "Research": 0, "Coding": 1, "Writing": 2, "General": 3
}

if st.button("Generate Prediction"):
    # Prepare the 9 features the model expects 
    # We include 'Impact_on_Grades' as a placeholder 0 because it was part of the fit 
    features = pd.DataFrame([[
        0, # Student_ID Placeholder
        age, 
        mapping.get(gender, 0), 
        mapping.get(edu, 0), 
        mapping.get(city, 0), 
        mapping.get(tool, 0), 
        usage, 
        mapping.get(purpose, 0), 
        0  # Impact_on_Grades Placeholder
    ]], columns=['Student_ID', 'Age', 'Gender', 'Education_Level', 'City', 
                 'AI_Tool_Used', 'Daily_Usage_Hours', 'Purpose', 'Impact_on_Grades'])

    try:
        prediction = model.predict(features)[0]
        
        # Display Results with Color Logic
        bg_color = "#d4edda" if prediction == "High" else "#fff3cd" if prediction == "Medium" else "#f8d7da"
        text_color = "#155724" if prediction == "High" else "#856404" if prediction == "Medium" else "#721c24"
        
        st.markdown(f"""
            <div class="result-card" style="background-color: {bg_color}; color: {text_color};">
                Predicted Grade Impact: {prediction}
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

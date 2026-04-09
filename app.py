import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Grade Impact Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for high-quality design
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Header Section
st.title("🎓 Student AI Impact Analyzer")
st.markdown("Analyze how AI tool usage influences academic performance based on student demographics and habits.")
st.divider()

# Input UI Design
st.subheader("📊 Enter Student Details")

# Organizing inputs into columns for a better look
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate", "PhD"])
    city = st.text_input("City", "New York")

with col2:
    ai_tool = st.selectbox("AI Tool Used", ["ChatGPT", "Gemini", "Claude", "Other"])
    usage_hours = st.slider("Daily Usage Hours", 0.0, 24.0, 2.0)
    purpose = st.selectbox("Primary Purpose", ["Research", "Writing", "Coding", "General Inquiry"])

# Create a placeholder for the dataframe
# Note: Ensure these values match the encoding used during model training
input_data = pd.DataFrame({
    'Student_ID': [0], # Placeholder ID
    'Age': [age],
    'Gender': [gender],
    'Education_Level': [education],
    'City': [city],
    'AI_Tool_Used': [ai_tool],
    'Daily_Usage_Hours': [usage_hours],
    'Purpose': [purpose]
})

# Prediction Logic
if st.button("Analyze Impact"):
    try:
        # Note: If your model requires encoded/numeric data, 
        # you must apply the same LabelEncoder/OneHotEncoder here before prediction.
        prediction = model.predict(input_data)
        result = prediction[0]
        
        st.divider()
        
        # Attractive result display
        if result == "High":
            color = "#D4EDDA"
            text_color = "#155724"
            icon = "🚀"
        elif result == "Medium":
            color = "#FFF3CD"
            text_color = "#856404"
            icon = "📈"
        else:
            color = "#F8D7DA"
            text_color = "#721C24"
            icon = "📉"
            
        st.markdown(f"""
            <div class="prediction-box" style="background-color: {color}; color: {text_color};">
                {icon} Predicted Impact on Grades: {result}
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}. Ensure inputs match the model's expected format.")

st.sidebar.info("This model uses K-Neighbors Classification to predict academic outcomes based on AI integration trends.")

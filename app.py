import streamlit as st
import pandas as pd
import joblib

# ... (Previous Styling Code) ...

def load_model():
    return joblib.load('model.pkl')

model = load_model()

# ... (Input UI Code) ...

# 1. DEFINE THE EXACT FEATURES THE MODEL EXPECTS 
# The model was trained with 'Impact_on_Grades' as a feature.
# We must include it in the dataframe to avoid the "Feature Names Missing" error.
input_data = pd.DataFrame({
    'Student_ID': [0],
    'Age': [age],
    'Gender': [gender],
    'Education_Level': [education],
    'City': [city],
    'AI_Tool_Used': [ai_tool],
    'Daily_Usage_Hours': [usage_hours],
    'Purpose': [purpose],
    'Impact_on_Grades': [0]  # Placeholder to satisfy the model's requirement 
})

# 2. ENSURE COLUMN ORDER 
# The model is sensitive to the order of these specific columns.
feature_order = [
    'Student_ID', 'Age', 'Gender', 'Education_Level', 'City', 
    'AI_Tool_Used', 'Daily_Usage_Hours', 'Purpose', 'Impact_on_Grades'
]
input_data = input_data[feature_order]

if st.button("Analyze Impact"):
    try:
        # Note: KNeighborsClassifier usually requires numeric input. 
        # If your model wasn't trained with a Pipeline, you may need to 
        # map these strings to numbers (e.g., Male -> 0, Female -> 1).
        prediction = model.predict(input_data)
        
        # ... (Rest of your display logic) ...
        st.success(f"The predicted impact level is: {prediction[0]}")
        
    except Exception as e:
        st.error(f"Error: {e}")

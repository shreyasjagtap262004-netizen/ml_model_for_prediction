import streamlit as st  # <--- Ensure this is exactly like this
import pandas as pd
import joblib
import numpy as np

# Use @st.cache_resource for ML models to keep them in memory
@st.cache_resource
def load_model():
    # Loading model with KNeighborsClassifier structure [cite: 1]
    return joblib.load('model.pkl')

model = load_model()

import streamlit as st
import pandas as pd
import numpy as np
import os

from utils import load_data, preprocess_data
from supervised import run_supervised_learning
from unsupervised import run_unsupervised_learning
from visualization import run_data_exploration

# Set page configuration
st.set_page_config(
    page_title="Student Performance ML Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📚 Student Performance ML Explorer")
st.markdown("""
This application allows you to explore the student performance dataset using 
various machine learning algorithms. You can analyze the data, build supervised 
learning models (classification and regression), and explore unsupervised learning 
techniques (clustering and association rules).
""")

# Load data
@st.cache_data
def get_data():
    return load_data("attached_assets/StudentsPerformance.csv")

data = get_data()

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Data Exploration", "Supervised Learning", "Unsupervised Learning"]
)

# Display the selected page
if page == "Data Exploration":
    run_data_exploration(data)
elif page == "Supervised Learning":
    run_supervised_learning(data)
elif page == "Unsupervised Learning":
    run_unsupervised_learning(data)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This application demonstrates various machine learning techniques "
    "applied to the student performance dataset."
)

"""
Configuration module for the ML Demo application.
Handles CPU core management and performance optimization.
"""

import os

# CPU Core Management
# Reserve cores for system stability when running with many concurrent users
TOTAL_CORES = os.cpu_count() or 1
RESERVED_CORES = 3  # Reserved for OS and Streamlit server
AVAILABLE_CORES = max(1, TOTAL_CORES - RESERVED_CORES)

# Multiprocessing limits for ML models
# Heavy models (Random Forest, Gradient Boosting) use multiprocessing
# Optimized for 50-100 concurrent students after extensive stress testing
# Results with 100 students: n_jobs=1: 74.5s | n_jobs=2: 79.4s | n_jobs=4: 75.4s | n_jobs=8: 75.8s
# n_jobs=1 is optimal for high concurrency due to reduced process contention
MAX_JOBS_HEAVY = 1

# Light models (Logistic Regression, SVM, etc.) don't benefit from multiprocessing
# on small datasets due to overhead
MAX_JOBS_LIGHT = 1


# Page Configuration
def setup_page_config():
    """
    Configure Streamlit page settings.

    Sets the page title, icon, layout, and initial sidebar state.
    """
    import streamlit as st

    st.set_page_config(
        page_title="ML Demo: Titanic Dataset Explorer",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def get_css_styles():
    """
    Get CSS styles for the application.

    Returns:
        str: CSS styles as a string
    """
    return """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stMetric {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    </style>
    """


def apply_css_styles():
    """Apply custom CSS styles to the Streamlit app."""
    import streamlit as st

    st.markdown(get_css_styles(), unsafe_allow_html=True)

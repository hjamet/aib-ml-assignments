"""
Configuration module for the ML Demo application.
Handles Streamlit configuration and CSS styles.
"""

import streamlit as st


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="ML Demo: Interactive Machine Learning",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def get_css_styles():
    """Return the CSS styles for the application."""
    return """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
"""


def apply_css_styles():
    """Apply CSS styles to the Streamlit app."""
    st.markdown(get_css_styles(), unsafe_allow_html=True)


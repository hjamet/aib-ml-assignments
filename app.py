"""
ML Demo: Interactive Machine Learning Application
Main entry point for the Streamlit application.
"""

import streamlit as st
import warnings

from src.utils.config import setup_page_config, apply_css_styles
from src.utils.data_loader import load_titanic_data
from src.utils.ui_components import display_dataset_overview
from src.pages.preprocessing_page import render_preprocessing_page
from src.pages.regression_page import render_regression_page
from src.pages.classification_page import render_classification_page

warnings.filterwarnings('ignore')


def main():
    """Main application function."""
    # Setup page configuration
    setup_page_config()
    
    # Apply CSS styles
    apply_css_styles()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Machine Learning Adventure: Interactive Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎓 Welcome to Your ML Journey!</h3>
        <p>This interactive demo will teach you Machine Learning concepts without any coding! 
        Simply use the controls in the sidebar to explore, learn, and experiment.</p>
        <p><strong>What you'll learn:</strong> Data exploration, preprocessing, model training, and evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_titanic_data()
    
    # Dataset overview
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    display_dataset_overview(df)
    
    # Show sample data
    if st.checkbox("📋 Show Sample Data", value=True):
        st.dataframe(df.head(10), width='stretch')
    
    # Sidebar navigation
    st.sidebar.markdown("# 🧭 Navigation")
    page = st.sidebar.radio(
        "Choisir une page:",
        ["📊 Preprocessing & Exploration", "📈 Régression", "🎯 Classification"],
        key="navigation_page"
    )
    
    # Render selected page
    if page == "📊 Preprocessing & Exploration":
        render_preprocessing_page(df)
    elif page == "📈 Régression":
        render_regression_page()
    elif page == "🎯 Classification":
        render_classification_page()
    
    # Learning Summary
    st.markdown('<h2 class="section-header">🎓 What You\'ve Learned</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <h3>🌟 Congratulations! You've mastered key ML concepts:</h3>
        <ul>
            <li><strong>Data Exploration:</strong> Understanding patterns in real-world data</li>
            <li><strong>Data Preprocessing:</strong> Preparing data for machine learning algorithms</li>
            <li><strong>Model Selection:</strong> Choosing the right algorithm for your problem</li>
            <li><strong>Model Evaluation:</strong> Understanding accuracy, precision, recall, and confusion matrices</li>
            <li><strong>Overfitting:</strong> Balancing model complexity for real-world performance</li>
            <li><strong>Feature Importance:</strong> Identifying which factors matter most</li>
        </ul>
        
        🚀 Real-world applications include:
        Medical diagnosis, fraud detection, recommendation systems, autonomous vehicles, 
        weather prediction, and much more!
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

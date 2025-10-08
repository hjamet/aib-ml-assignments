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


def initialize_session_state():
    """Initialize session_state with default values for all widgets."""
    # Preprocessing page defaults
    if "exploration_feature" not in st.session_state:
        st.session_state.exploration_feature = "sex"
    if "missing_age_option" not in st.session_state:
        st.session_state.missing_age_option = "Fill with median"
    if "normalize_features" not in st.session_state:
        st.session_state.normalize_features = True
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = ["Age", "Sex", "Passenger Class", "Fare"]
    if "test_size" not in st.session_state:
        st.session_state.test_size = 20
    
    # Regression page defaults
    if "regression_models" not in st.session_state:
        st.session_state.regression_models = ["Linear Regression", "Random Forest", "Ridge Regression"]
    
    # Classification page defaults
    if "classification_models" not in st.session_state:
        st.session_state.classification_models = ["Random Forest", "Logistic Regression", "Support Vector Machine"]


def home_page():
    """Home page with dataset overview."""
    # Load data
    df = load_titanic_data()
    
    # Store in session state for other pages
    st.session_state.df = df
    
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
    
    # Dataset overview
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    display_dataset_overview(df)
    
    # Show sample data
    if st.checkbox("📋 Show Sample Data", value=True, key="home_show_sample"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Learning Summary
    st.markdown('<h2 class="section-header">🎓 What You\'ll Learn</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <h3>🌟 Master key ML concepts:</h3>
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


def preprocessing_page_wrapper():
    """Wrapper for preprocessing page."""
    df = st.session_state.get('df', load_titanic_data())
    render_preprocessing_page(df)


def main():
    """Main application function."""
    # Setup page configuration
    setup_page_config()
    
    # Apply CSS styles
    apply_css_styles()
    
    # Initialize session state
    initialize_session_state()
    
    # Create navigation with st.Page
    pg = st.navigation([
        st.Page(home_page, title="Home", icon="🏠", default=True),
        st.Page(preprocessing_page_wrapper, title="Preprocessing & Exploration", icon="📊"),
        st.Page(render_regression_page, title="Regression", icon="📈"),
        st.Page(render_classification_page, title="Classification", icon="🎯"),
    ])
    
    # Run the selected page
    pg.run()


if __name__ == "__main__":
    main()

"""
Preprocessing and Exploration page for the ML Demo application.
"""

import streamlit as st
import plotly.express as px

from ..utils.preprocessing import preprocess_data
from ..utils.ui_components import display_preprocessing_results


def render_preprocessing_page(df):
    """
    Render the preprocessing and exploration page.
    
    Args:
        df (pd.DataFrame): The Titanic dataset
    """
    st.markdown('<h2 class="section-header">🔍 Interactive Data Exploration</h2>', unsafe_allow_html=True)
    
    exploration_feature = st.selectbox(
        "🔍 Explore Feature:",
        ["sex", "pclass", "age", "fare", "embarked"],
        format_func=lambda x: {
            "sex": "👥 Gender", 
            "pclass": "🎫 Passenger Class",
            "age": "👶 Age",
            "fare": "💰 Fare",
            "embarked": "🚢 Port of Embarkation"
        }[x],
        key="exploration_feature"
    )
    
    # Create exploration plots
    col1, col2 = st.columns(2)
    
    with col1:
        if exploration_feature in ['age', 'fare']:
            fig = px.histogram(df, x=exploration_feature, title=f'Distribution of {exploration_feature.title()}',
                             color_discrete_sequence=['skyblue'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(df, x=exploration_feature, title=f'Distribution of {exploration_feature.title()}',
                             color_discrete_sequence=['lightgreen'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        survival_by_feature = df.groupby(exploration_feature)['survived'].mean().reset_index()
        fig = px.bar(survival_by_feature, x=exploration_feature, y='survived',
                    title=f'Survival Rate by {exploration_feature.title()}',
                    color_discrete_sequence=['orange'])
        fig.update_yaxes(title='Survival Rate')
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    insights = {
        'sex': "👩‍👩‍👧‍👦 Women had much higher survival rates than men ('Women and children first!')",
        'pclass': "🥇 First-class passengers had better survival chances than lower classes",
        'age': "👶 Children and young adults generally had better survival chances",
        'fare': "💎 Passengers who paid higher fares (likely in better cabins) survived more often",
        'embarked': "🚢 The port of embarkation might indicate passenger class or cabin location"
    }
    
    st.info(f"💡 **Insight:** {insights[exploration_feature]}")
    
    # Section 2: Data Preprocessing
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Preprocessing controls
    st.selectbox(
        "Handle Missing Ages:",
        ["Fill with median", "Fill with mean", "Drop rows"],
        key="missing_age_option"
    )
    
    st.checkbox("📏 Normalize Features", value=True, key="normalize_features")
    
    st.multiselect(
        "📊 Select Features:",
        ["Age", "Sex", "Passenger Class", "Fare", "Siblings/Spouses", "Parents/Children", "Port of Embarkation"],
        default=["Age", "Sex", "Passenger Class", "Fare"],
        key="selected_features"
    )
    
    st.slider("Test Set Size (%)", 10, 40, 20, 5, key="test_size")
    
    if st.session_state.selected_features:
        # Get values from session_state (set by widgets)
        selected_features = st.session_state.selected_features
        normalize_features = st.session_state.normalize_features
        missing_age_option = st.session_state.missing_age_option
        
        # Preprocess data
        X, y, feature_names, scaler, before_count, after_count = preprocess_data(
            df, selected_features, missing_age_option, normalize_features
        )
        
        # Store processed data in session state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.feature_names = feature_names
        st.session_state.scaler = scaler
        st.session_state.selected_features_persistent = selected_features
        
        # Show preprocessing results
        display_preprocessing_results(before_count, after_count, len(selected_features), selected_features)


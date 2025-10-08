"""
Preprocessing and Exploration page for the ML Demo application.
"""

import streamlit as st
import plotly.express as px

from ..utils.preprocessing import preprocess_data
from ..utils.ui_components import display_preprocessing_results, display_dataset_overview


def save_to_state(temp_key, perm_key):
    """Save widget value from temporary key to permanent key."""
    st.session_state[perm_key] = st.session_state[temp_key]


def render_preprocessing_page(df):
    """
    Render the preprocessing and exploration page.
    
    Args:
        df (pd.DataFrame): The Titanic dataset
    """
    # Dataset overview
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    display_dataset_overview(df)
    
    # Show sample data
    if st.checkbox("📋 Show Sample Data", value=True, key="preprocessing_show_sample"):
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown('<h2 class="section-header">🔍 Interactive Data Exploration</h2>', unsafe_allow_html=True)
    
    feature_options = ["sex", "pclass", "age", "fare", "embarked"]
    feature_labels = {
        "sex": "👥 Gender", 
        "pclass": "🎫 Passenger Class",
        "age": "👶 Age",
        "fare": "💰 Fare",
        "embarked": "🚢 Port of Embarkation"
    }
    
    # Initialize temp key from permanent key if needed
    if "_exploration_feature" not in st.session_state:
        st.session_state["_exploration_feature"] = st.session_state.get("exploration_feature", "sex")
    
    exploration_feature = st.selectbox(
        "🔍 Explore Feature:",
        feature_options,
        index=feature_options.index(st.session_state["_exploration_feature"]),
        format_func=lambda x: feature_labels[x],
        key="_exploration_feature",
        on_change=save_to_state,
        args=("_exploration_feature", "exploration_feature")
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
    # Initialize temp keys from permanent keys if needed
    if "_missing_age_option" not in st.session_state:
        st.session_state["_missing_age_option"] = st.session_state.get("missing_age_option", "Fill with median")
    if "_normalize_features" not in st.session_state:
        st.session_state["_normalize_features"] = st.session_state.get("normalize_features", True)
    if "_selected_features" not in st.session_state:
        st.session_state["_selected_features"] = st.session_state.get("selected_features", ["Age", "Sex", "Passenger Class", "Fare"])
    if "_test_size" not in st.session_state:
        st.session_state["_test_size"] = st.session_state.get("test_size", 20)
    
    missing_age_options = ["Fill with median", "Fill with mean", "Drop rows"]
    st.selectbox(
        "Handle Missing Ages:",
        missing_age_options,
        index=missing_age_options.index(st.session_state["_missing_age_option"]),
        key="_missing_age_option",
        on_change=save_to_state,
        args=("_missing_age_option", "missing_age_option")
    )
    
    st.checkbox(
        "📏 Normalize Features",
        value=st.session_state["_normalize_features"],
        key="_normalize_features",
        on_change=save_to_state,
        args=("_normalize_features", "normalize_features")
    )
    
    feature_options = ["Age", "Sex", "Passenger Class", "Fare", "Siblings/Spouses", "Parents/Children", "Port of Embarkation"]
    st.multiselect(
        "📊 Select Features:",
        feature_options,
        default=st.session_state["_selected_features"],
        key="_selected_features",
        on_change=save_to_state,
        args=("_selected_features", "selected_features")
    )
    
    st.slider(
        "Test Set Size (%)",
        10, 40,
        value=st.session_state["_test_size"],
        step=5,
        key="_test_size",
        on_change=save_to_state,
        args=("_test_size", "test_size")
    )
    
    if st.session_state.get("selected_features"):
        # Get values from permanent session_state keys
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


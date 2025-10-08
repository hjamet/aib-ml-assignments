"""
Preprocessing and Exploration page for the ML Demo application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from ..utils.preprocessing import preprocess_data, get_feature_mapping, get_reverse_feature_mapping
from ..utils.ui_components import display_preprocessing_results, display_dataset_overview
from ..utils.toc import init_toc, render_toc, toc_markdown, toc_subheader


def save_to_state(temp_key, perm_key):
    """Save widget value from temporary key to permanent key."""
    st.session_state[perm_key] = st.session_state[temp_key]


def render_preprocessing_page(df):
    """
    Render the preprocessing and exploration page.
    
    Args:
        df (pd.DataFrame): The Titanic dataset
    """
    # Initialize TOC
    init_toc()
    
    # ===== CONFIGURATION SECTION =====
    toc_markdown('<h2 class="section-header">🔧 Configuration</h2>', level=1, unsafe_allow_html=True)
    
    # Target variable selection
    target_options = list(df.columns)
    if "_target_column" not in st.session_state:
        st.session_state["_target_column"] = st.session_state.get("target_column", "survived")
    
    # Validate target is still in columns
    if st.session_state["_target_column"] not in target_options:
        st.session_state["_target_column"] = "survived"
    
    target_column = st.selectbox(
        "🎯 Target Variable (to predict):",
        target_options,
        index=target_options.index(st.session_state["_target_column"]),
        key="_target_column",
        on_change=save_to_state,
        args=("_target_column", "target_column"),
        help="Select which variable you want to predict"
    )
    
    # Warning for non-numeric targets
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        st.warning(f"⚠️ '{target_column}' is categorical. It will be encoded for Classification models.")
    
    # Categorical encoding method
    encoding_options = ["Drop Columns", "Label Encoding", "One-Hot Encoding"]
    if "_encoding_method" not in st.session_state:
        st.session_state["_encoding_method"] = st.session_state.get("encoding_method", "Drop Columns")
    
    st.selectbox(
        "🔤 Categorical Features Encoding:",
        encoding_options,
        index=encoding_options.index(st.session_state["_encoding_method"]),
        key="_encoding_method",
        on_change=save_to_state,
        args=("_encoding_method", "encoding_method"),
        help="How to handle text/categorical columns: sex, embarked, class, deck, etc."
    )
    
    # Preprocessing controls
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
    
    # Feature selection - exclude target column
    feature_options = ["Age", "Sex", "Passenger Class", "Fare", "Siblings/Spouses", "Parents/Children", "Port of Embarkation", "Survived"]
    feature_mapping = get_feature_mapping()
    reverse_feature_mapping = get_reverse_feature_mapping()
    
    # Exclure la target des features sélectionnables
    # Check if the target_column corresponds to any feature in feature_options
    target_feature_name = reverse_feature_mapping.get(st.session_state.target_column)
    available_features = [f for f in feature_options 
                         if f != target_feature_name and feature_mapping.get(f) != st.session_state.target_column]
    
    # Validate selected features
    valid_selected = [f for f in st.session_state["_selected_features"] if f in available_features]
    if not valid_selected:
        valid_selected = [f for f in ["Age", "Sex", "Passenger Class", "Fare"] if f in available_features]
    st.session_state["_selected_features"] = valid_selected
    
    st.multiselect(
        "📊 Select Features:",
        available_features,
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
    
    # ===== TWO-COLUMN LAYOUT =====
    toc_markdown('<h2 class="section-header">📊 Data Comparison</h2>', level=1, unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    # ===== LEFT COLUMN: ORIGINAL DATA =====
    with col_left:
        toc_subheader("📄 Original Data")
        
        # Dataset overview
        st.markdown("**Dataset Metrics:**")
        display_dataset_overview(df)
        
        # Sample data with target highlighted
        st.markdown(f"**🎯 Target Variable: `{st.session_state.target_column}`**")
        # Try to highlight with emoji in column name
        df_display = df.copy()
        target_col = st.session_state.target_column
        df_display = df_display.rename(columns={target_col: f"🎯 {target_col}"})
        st.dataframe(df_display.head(10), use_container_width=True)
        
        # Interactive Exploration
        st.markdown("**Data Exploration:**")
        
        feature_options_explore = ["sex", "pclass", "age", "fare", "embarked"]
        feature_labels = {
            "sex": "👥 Gender", 
            "pclass": "🎫 Passenger Class",
            "age": "👶 Age",
            "fare": "💰 Fare",
            "embarked": "🚢 Port of Embarkation"
        }
        
        # Initialize temp key from permanent key if needed
        if "_exploration_feature_left" not in st.session_state:
            st.session_state["_exploration_feature_left"] = st.session_state.get("exploration_feature_left", "sex")
        
        # Validate
        if st.session_state["_exploration_feature_left"] not in feature_options_explore:
            st.session_state["_exploration_feature_left"] = "sex"
        
        exploration_feature_left = st.selectbox(
            "Explore Feature:",
            feature_options_explore,
            index=feature_options_explore.index(st.session_state["_exploration_feature_left"]),
            format_func=lambda x: feature_labels[x],
            key="_exploration_feature_left",
            on_change=save_to_state,
            args=("_exploration_feature_left", "exploration_feature_left")
        )
        
        # Distribution plot
        if exploration_feature_left in ['age', 'fare']:
            fig = px.histogram(df, x=exploration_feature_left, 
                             title=f'Distribution of {exploration_feature_left.title()}',
                             color_discrete_sequence=['skyblue'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(df, x=exploration_feature_left, 
                             title=f'Distribution of {exploration_feature_left.title()}',
                             color_discrete_sequence=['lightgreen'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Survival rate plot (if survived column exists)
        if 'survived' in df.columns:
            survival_by_feature = df.groupby(exploration_feature_left)['survived'].mean().reset_index()
            fig = px.bar(survival_by_feature, x=exploration_feature_left, y='survived',
                        title=f'Survival Rate by {exploration_feature_left.title()}',
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
        
        st.info(f"💡 **Insight:** {insights.get(exploration_feature_left, 'Explore patterns in this feature!')}")
    
    # ===== RIGHT COLUMN: TRANSFORMED DATA =====
    with col_right:
        toc_subheader("⚙️ Transformed Data")
        
        if st.session_state.get("selected_features"):
            # Get values from permanent session_state keys
            selected_features = st.session_state.selected_features
            normalize_features = st.session_state.normalize_features
            missing_age_option = st.session_state.missing_age_option
            target_column_val = st.session_state.target_column
            encoding_method_val = st.session_state.encoding_method
            
            # Preprocess data
            X, y, feature_names, scaler, before_count, after_count, df_transformed = preprocess_data(
                df, selected_features, missing_age_option, normalize_features,
                target_column=target_column_val, encoding_method=encoding_method_val
            )
            
            # Store processed data in session state
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.feature_names = feature_names
            st.session_state.scaler = scaler
            st.session_state.selected_features_persistent = selected_features
            st.session_state.target_column_persistent = target_column_val
            st.session_state.encoding_method_persistent = encoding_method_val
            st.session_state.df_transformed = df_transformed
            
            # Show preprocessing results
            st.markdown("**Preprocessing Results:**")
            display_preprocessing_results(before_count, after_count, len(selected_features), selected_features)
            
            # Sample transformed data with target highlighted
            st.markdown(f"**🎯 Target Variable: `{target_column_val}`**")
            df_transformed_display = df_transformed.copy()
            if target_column_val in df_transformed_display.columns:
                df_transformed_display = df_transformed_display.rename(columns={target_column_val: f"🎯 {target_column_val}"})
            st.dataframe(df_transformed_display.head(10), use_container_width=True)
            
            # Interactive Exploration on transformed data
            st.markdown("**Data Exploration:**")
            
            # Get available numeric features for exploration
            numeric_features = [col for col in df_transformed.columns if col != target_column_val]
            
            if numeric_features:
                # Create feature labels with emojis for transformed features
                transformed_feature_labels = {}
                for feature in numeric_features:
                    feature_lower = feature.lower()
                    # Map transformed column names to emoji labels
                    if 'age' in feature_lower:
                        transformed_feature_labels[feature] = f"👶 {feature.replace('_', ' ').title()}"
                    elif 'sex' in feature_lower:
                        transformed_feature_labels[feature] = f"👥 {feature.replace('_', ' ').title()}"
                    elif 'pclass' in feature_lower or 'class' in feature_lower:
                        transformed_feature_labels[feature] = f"🎫 {feature.replace('_', ' ').title()}"
                    elif 'fare' in feature_lower:
                        transformed_feature_labels[feature] = f"💰 {feature.replace('_', ' ').title()}"
                    elif 'sibsp' in feature_lower:
                        transformed_feature_labels[feature] = f"👨‍👩‍👧 {feature.replace('_', ' ').title()}"
                    elif 'parch' in feature_lower:
                        transformed_feature_labels[feature] = f"👨‍👩‍👦 {feature.replace('_', ' ').title()}"
                    elif 'embarked' in feature_lower:
                        transformed_feature_labels[feature] = f"🚢 {feature.replace('_', ' ').title()}"
                    elif 'deck' in feature_lower:
                        transformed_feature_labels[feature] = f"🏢 {feature.replace('_', ' ').title()}"
                    elif 'alone' in feature_lower:
                        transformed_feature_labels[feature] = f"🧍 {feature.replace('_', ' ').title()}"
                    else:
                        transformed_feature_labels[feature] = feature.replace('_', ' ').title()
                
                # Initialize temp key
                if "_exploration_feature_right" not in st.session_state:
                    st.session_state["_exploration_feature_right"] = st.session_state.get("exploration_feature_right", numeric_features[0])
                
                # Validate
                if st.session_state["_exploration_feature_right"] not in numeric_features:
                    st.session_state["_exploration_feature_right"] = numeric_features[0]
                
                exploration_feature_right = st.selectbox(
                    "Explore Feature:",
                    numeric_features,
                    index=numeric_features.index(st.session_state["_exploration_feature_right"]),
                    format_func=lambda x: transformed_feature_labels.get(x, x),
                    key="_exploration_feature_right",
                    on_change=save_to_state,
                    args=("_exploration_feature_right", "exploration_feature_right")
                )
                
                # Distribution plot
                fig = px.histogram(df_transformed, x=exploration_feature_right, 
                                 title=f'Distribution of {exploration_feature_right}',
                                 color_discrete_sequence=['purple'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Target correlation plot - Bar chart showing target rate by feature
                if target_column_val in df_transformed.columns:
                    target_by_feature = df_transformed.groupby(exploration_feature_right)[target_column_val].mean().reset_index()
                    fig = px.bar(target_by_feature, x=exploration_feature_right, y=target_column_val,
                                title=f'{target_column_val.title()} Rate by {exploration_feature_right}',
                                color_discrete_sequence=['orange'])
                    fig.update_yaxes(title=f'{target_column_val.title()} Rate')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Insights for transformed features (same as original)
                # Match by feature name pattern to handle encoded columns
                feature_lower = exploration_feature_right.lower()
                
                if 'sex' in feature_lower:
                    insight_msg = "👩‍👩‍👧‍👦 Women had much higher survival rates than men ('Women and children first!')"
                elif 'pclass' in feature_lower or 'class' in feature_lower:
                    insight_msg = "🥇 First-class passengers had better survival chances than lower classes"
                elif 'age' in feature_lower:
                    insight_msg = "👶 Children and young adults generally had better survival chances"
                elif 'fare' in feature_lower:
                    insight_msg = "💎 Passengers who paid higher fares (likely in better cabins) survived more often"
                elif 'embarked' in feature_lower:
                    insight_msg = "🚢 The port of embarkation might indicate passenger class or cabin location"
                elif 'sibsp' in feature_lower:
                    insight_msg = "👨‍👩‍👧 Number of siblings/spouses aboard affects survival patterns"
                elif 'parch' in feature_lower:
                    insight_msg = "👨‍👩‍👦 Number of parents/children aboard can indicate family groups"
                elif 'deck' in feature_lower:
                    insight_msg = "🏢 Deck location often correlates with passenger class and proximity to lifeboats"
                elif 'alone' in feature_lower:
                    insight_msg = "🧍 Traveling alone or with family significantly affected survival chances"
                else:
                    insight_msg = "Explore patterns in this transformed feature!"
                
                st.info(f"💡 **Insight:** {insight_msg}")
            else:
                st.info("No features selected for transformation.")
        else:
            st.info("⚠️ Please select features in the configuration section above to see transformed data.")
    
    # Render TOC at the end
    render_toc()

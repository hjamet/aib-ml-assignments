"""
Regression page for the ML Demo application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..utils.model_factory import get_available_models
from ..utils.model_training import train_models
from ..utils.visualization import (
    create_train_metrics_chart, create_test_metrics_chart,
    create_regression_surface_plot, create_2d_scatter_plot, 
    create_feature_importance_plot, create_feature_coefficients_plot
)
from ..utils.ui_components import (
    display_model_metrics_columns, display_metrics_table, render_hyperparameter_controls,
    render_prediction_inputs, display_prediction_result
)


def save_to_state(temp_key, perm_key):
    """Save widget value from temporary key to permanent key."""
    st.session_state[perm_key] = st.session_state[temp_key]


def render_regression_page():
    """Render the regression page."""
    # Check if preprocessing is done
    if 'X' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord configurer le preprocessing dans l'onglet 'Preprocessing & Exploration'")
        st.stop()
    
    # Retrieve preprocessed data
    X = st.session_state.X
    y = st.session_state.y
    feature_names = st.session_state.feature_names
    scaler = st.session_state.scaler
    normalize_features = st.session_state.get('normalize_features', True)
    selected_features = st.session_state.get('selected_features_persistent', [])
    
    st.markdown('<h2 class="section-header">🤖 Regression Models Training & Comparison</h2>', unsafe_allow_html=True)
    
    # Model selection in page
    st.markdown("### 📋 Model Selection")
    
    available_models = get_available_models("Regression")
    
    # Initialize temp key from permanent key if needed
    if "_regression_models" not in st.session_state:
        st.session_state["_regression_models"] = st.session_state.get("regression_models", ["Linear Regression", "Random Forest", "Ridge Regression"])
    
    selected_models = st.multiselect(
        "Choose Regression Models:",
        list(available_models.keys()),
        default=st.session_state["_regression_models"],
        help="Select multiple models to compare their performance",
        key="_regression_models",
        on_change=save_to_state,
        args=("_regression_models", "regression_models")
    )
    
    # Metric selection
    st.markdown("### 📊 Metric Selection")
    
    available_metrics = ['R²', 'RMSE', 'MAE']
    
    # Initialize temp key from permanent key if needed
    if "_regression_metrics" not in st.session_state:
        st.session_state["_regression_metrics"] = st.session_state.get("regression_metrics", available_metrics)
    
    selected_metrics = st.multiselect(
        "Choose Metrics to Display:",
        available_metrics,
        default=st.session_state["_regression_metrics"],
        help="Select which metrics to display in tables and charts",
        key="_regression_metrics",
        on_change=save_to_state,
        args=("_regression_metrics", "regression_metrics")
    )
    
    # Get test size from session state
    test_size = st.session_state.get("test_size", 20)
    
    # Hyperparameters in page
    st.markdown("### ⚙️ Hyperparameter Tuning")
    st.info("💡 Hyperparameters control how models learn. Experiment with different values!")
    hyperparams = {}
    
    for model_name in selected_models:
        with st.expander(f"🔧 {model_name} - Hyperparameters"):
            hyperparams[model_name] = render_hyperparameter_controls(model_name, "Regression", "reg")
    
    if selected_models:
        # Train models
        models, results, X_train, X_test, y_train, y_test = train_models(
            X, y, selected_models, hyperparams, test_size, "Regression"
        )
        
        # Display results
        results_df = pd.DataFrame(results)
        
        st.subheader("📊 Model Performance Comparison")
        
        # Metrics display as table
        display_metrics_table(results_df, "Regression", selected_metrics)
        
        # Comparison charts in two columns
        st.markdown("### 📈 Performance Metrics Comparison")
        col_train, col_test = st.columns(2)
        
        with col_train:
            fig_train = create_train_metrics_chart(results_df, "Regression", selected_metrics)
            st.plotly_chart(fig_train, use_container_width=True)
        
        with col_test:
            fig_test = create_test_metrics_chart(results_df, "Regression", selected_metrics)
            st.plotly_chart(fig_test, use_container_width=True)
        
        # Model Exploration - Allow student to select any trained model
        st.markdown("### 🔍 Model Exploration")
        st.info("💡 Select a model below to explore its detailed results and performance metrics.")
        
        # Initialize temp key from permanent key if needed
        if "_selected_exploration_model_regression" not in st.session_state:
            default_model = st.session_state.get("selected_exploration_model_regression", selected_models[0])
            if default_model not in selected_models:
                default_model = selected_models[0]
            st.session_state["_selected_exploration_model_regression"] = default_model
        
        # Validate that saved value is still valid
        if st.session_state["_selected_exploration_model_regression"] not in selected_models:
            st.session_state["_selected_exploration_model_regression"] = selected_models[0]
        
        selected_model_name = st.selectbox(
            "Choose a model to explore:",
            selected_models,
            index=selected_models.index(st.session_state["_selected_exploration_model_regression"]),
            key="_selected_exploration_model_regression",
            on_change=save_to_state,
            args=("_selected_exploration_model_regression", "selected_exploration_model_regression")
        )
        
        selected_model = models[selected_model_name]
        y_pred_selected = selected_model.predict(X_test)
        
        # Regression visualizations
        st.markdown(f"### 📈 Regression Analysis - {selected_model_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(x=y_test, y=y_pred_selected, 
                           title=f"Actual vs Predicted Values - {selected_model_name}",
                           labels={'x': 'Actual Values', 'y': 'Predicted Values'})
            # Add perfect prediction line
            min_val, max_val = min(y_test.min(), y_pred_selected.min()), max(y_test.max(), y_pred_selected.max())
            fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                        line=dict(color="red", width=2, dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals plot
            residuals = y_test - y_pred_selected
            fig = px.scatter(x=y_pred_selected, y=residuals,
                           title=f"Residuals Plot - {selected_model_name}",
                           labels={'x': 'Predicted Values', 'y': 'Residuals'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics explanation
        mse = mean_squared_error(y_test, y_pred_selected)
        mae = mean_absolute_error(y_test, y_pred_selected)
        r2 = r2_score(y_test, y_pred_selected)
        
        st.markdown(f"""
        **📊 Regression Metrics for {selected_model_name}:**
        - **R² Score: {r2:.3f}** - Proportion of variance explained (1.0 = perfect fit)
        - **RMSE: {np.sqrt(mse):.2f}** - Average prediction error in original units
        - **MAE: {mae:.2f}** - Mean absolute error (robust to outliers)
        """)
        
        # 2D Visualization
        if len(feature_names) >= 2:
            st.subheader(f"📊 2D Feature Visualization - {selected_model_name}")
            
            col1, col2 = st.columns(2)
            with col1:
                # Initialize temp key from permanent key if needed
                if "_reg_2d_x" not in st.session_state:
                    default_x = st.session_state.get("reg_2d_x", feature_names[0])
                    if default_x not in feature_names:
                        default_x = feature_names[0]
                    st.session_state["_reg_2d_x"] = default_x
                
                # Validate that saved value is still valid
                if st.session_state["_reg_2d_x"] not in feature_names:
                    st.session_state["_reg_2d_x"] = feature_names[0]
                
                feature_x = st.selectbox(
                    "Choose X-axis feature:",
                    feature_names,
                    index=feature_names.index(st.session_state["_reg_2d_x"]),
                    key="_reg_2d_x",
                    on_change=save_to_state,
                    args=("_reg_2d_x", "reg_2d_x")
                )
            with col2:
                available_y = [f for f in feature_names if f != feature_x]
                
                # Initialize temp key from permanent key if needed
                if "_reg_2d_y" not in st.session_state:
                    default_y = st.session_state.get("reg_2d_y", available_y[0] if available_y else feature_names[0])
                    if default_y not in available_y:
                        default_y = available_y[0] if available_y else feature_names[0]
                    st.session_state["_reg_2d_y"] = default_y
                
                # Validate that saved value is still valid
                if st.session_state["_reg_2d_y"] not in available_y:
                    st.session_state["_reg_2d_y"] = available_y[0] if available_y else feature_names[0]
                
                feature_y = st.selectbox(
                    "Choose Y-axis feature:",
                    available_y,
                    index=available_y.index(st.session_state["_reg_2d_y"]),
                    key="_reg_2d_y",
                    on_change=save_to_state,
                    args=("_reg_2d_y", "reg_2d_y")
                )
            
            if len(feature_names) == 2:
                fig = create_regression_surface_plot(X_train, y_train, selected_model, feature_x, feature_y, selected_model_name, "Survived")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"💡 **Prediction Surface for {selected_model_name}**: The colored surface shows the model's predictions across the feature space.")
            else:
                fig = create_2d_scatter_plot(X_test, y_test, y_pred_selected, feature_x, feature_y, "Regression", "Survived")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"💡 **2D Feature Plot for {selected_model_name}**: Each point is a passenger. Color intensity shows the target value. Size shows prediction accuracy.")
        
        # Feature Importance
        if hasattr(selected_model, 'feature_importances_'):
            st.subheader(f"🎯 Feature Importance - {selected_model_name}")
            
            fig = create_feature_importance_plot(feature_names, selected_model.feature_importances_, selected_model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Feature Importance for {selected_model_name}**: Shows which passenger characteristics the model considers most important.")
        
        elif hasattr(selected_model, 'coef_'):
            st.subheader(f"🎯 Feature Coefficients - {selected_model_name}")
            
            fig = create_feature_coefficients_plot(feature_names, selected_model.coef_, selected_model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Feature Coefficients for {selected_model_name}**: Shows how much each feature influences the model's predictions.")
        
        # Interactive Prediction
        st.markdown('<h2 class="section-header">🔮 Make Your Own Predictions</h2>', unsafe_allow_html=True)
        
        st.markdown(f"Try different passenger profiles and see what **{selected_model_name}** predicts!")
        
        col1, col2 = st.columns(2)
        
        # Initialize prediction_inputs before columns to avoid UnboundLocalError
        prediction_inputs = {}
        
        with col1:
            st.subheader("👤 Passenger Profile")
            
            if not feature_names:
                st.warning("⚠️ Please select features in the 'Preprocessing & Exploration' page first to make predictions.")
            else:
                prediction_inputs = render_prediction_inputs(selected_features, "reg")
        
        with col2:
            st.subheader("🎯 Prediction Result")
            
            if selected_features and st.button("🔮 Predict Value", type="primary", key="reg_predict_btn"):
                input_data = []
                for feature_name in feature_names:
                    if feature_name in prediction_inputs:
                        input_data.append(prediction_inputs[feature_name])
                    else:
                        input_data.append(X[feature_name].median())
                
                input_array = np.array(input_data).reshape(1, -1)
                
                if normalize_features and scaler is not None:
                    input_array = scaler.transform(input_array)
                
                prediction = selected_model.predict(input_array)[0]
                
                display_prediction_result(prediction, None, y, "Regression")


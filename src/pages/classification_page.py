"""
Classification page for the ML Demo application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from ..utils.model_factory import get_available_models
from ..utils.model_training import train_models
from ..utils.visualization import (
    create_train_metrics_chart, create_test_metrics_chart,
    create_decision_boundary_plot, create_2d_scatter_plot, 
    create_confusion_matrix_plot, create_confusion_matrix_train_plot,
    create_feature_importance_plot, create_feature_coefficients_plot
)
from ..utils.ui_components import (
    display_model_metrics_columns, display_metrics_table, render_hyperparameter_controls,
    render_prediction_inputs, display_prediction_result, inject_card_styles,
    render_model_info_cards, render_metric_info_cards
)
from ..utils.toc import init_toc, render_toc, toc_markdown, toc_header, toc_subheader, toc_subsubheader


def save_to_state(temp_key, perm_key):
    """Save widget value from temporary key to permanent key."""
    st.session_state[perm_key] = st.session_state[temp_key]


def render_classification_page():
    """Render the classification page."""
    # Initialize TOC
    init_toc()
    
    # Inject card styles
    inject_card_styles()
    
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
    
    # Check if target variable is continuous (more than 10 unique values)
    n_unique = y.nunique()
    if n_unique > 10:
        target_name = st.session_state.get('target_column_persistent', 'target')
        st.error(f"⚠️ **This page cannot be used when working with continuous labels.**\n\n"
                 f"You are trying to predict `{target_name}`, which has {n_unique} unique values and appears to be a continuous variable.\n\n"
                 f"**Please use the Regression page instead**, as the Classification page is designed for categorical target variables.")
        st.stop()
    
    toc_markdown('<h2 class="section-header">🤖 Classification Models Training & Comparison</h2>', level=1, unsafe_allow_html=True)
    
    # Model selection in page
    toc_subheader("📋 Model Selection")
    
    available_models = get_available_models("Classification")
    
    # Initialize temp key from permanent key if needed
    if "_classification_models" not in st.session_state:
        st.session_state["_classification_models"] = st.session_state.get("classification_models", ["Random Forest", "Logistic Regression", "Support Vector Machine"])
    
    selected_models = st.multiselect(
        "Choose Classification Models:",
        list(available_models.keys()),
        default=st.session_state["_classification_models"],
        help="Select multiple models to compare their performance",
        key="_classification_models",
        on_change=save_to_state,
        args=("_classification_models", "classification_models")
    )
    
    # Display model information cards
    if selected_models:
        toc_subsubheader("🎓 Learn About Your Selected Models")
        render_model_info_cards(selected_models, "Classification")
    
    # Metric selection
    toc_subheader("📊 Metric Selection")
    
    available_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    # Initialize temp key from permanent key if needed
    if "_classification_metrics" not in st.session_state:
        st.session_state["_classification_metrics"] = st.session_state.get("classification_metrics", available_metrics)
    
    selected_metrics = st.multiselect(
        "Choose Metrics to Display:",
        available_metrics,
        default=st.session_state["_classification_metrics"],
        help="Select which metrics to display in tables and charts",
        key="_classification_metrics",
        on_change=save_to_state,
        args=("_classification_metrics", "classification_metrics")
    )
    
    # Display metric information cards
    if selected_metrics:
        toc_subsubheader("📚 Understanding Your Selected Metrics")
        render_metric_info_cards(selected_metrics, "Classification")
    
    # Get test size from session state
    test_size = st.session_state.get("test_size", 20)
    
    # Hyperparameters in page
    toc_subheader("⚙️ Hyperparameter Tuning")
    st.info("💡 Hyperparameters control how models learn. Experiment with different values!")
    hyperparams = {}
    
    for model_name in selected_models:
        with st.expander(f"🔧 {model_name} - Hyperparameters"):
            hyperparams[model_name] = render_hyperparameter_controls(model_name, "Classification", "clf")
    
    if selected_models:
        # Train models
        models, results, X_train, X_test, y_train, y_test = train_models(
            X, y, selected_models, hyperparams, test_size, "Classification"
        )
        
        # Display results
        results_df = pd.DataFrame(results)
        
        toc_subheader("📊 Model Performance Comparison")
        
        # Metrics display as table
        display_metrics_table(results_df, "Classification", selected_metrics)
        
        # Comparison charts in two columns
        toc_subheader("📈 Performance Metrics Comparison")
        col_train, col_test = st.columns(2)
        
        with col_train:
            fig_train = create_train_metrics_chart(results_df, "Classification", selected_metrics)
            st.plotly_chart(fig_train, use_container_width=True)
        
        with col_test:
            fig_test = create_test_metrics_chart(results_df, "Classification", selected_metrics)
            st.plotly_chart(fig_test, use_container_width=True)
        
        # Model Exploration - Allow student to select any trained model
        toc_subheader("🔍 Model Exploration")
        st.info("💡 Select a model below to explore its detailed results and performance metrics.")
        
        # Initialize temp key from permanent key if needed
        if "_selected_exploration_model_classification" not in st.session_state:
            default_model = st.session_state.get("selected_exploration_model_classification", selected_models[0])
            if default_model not in selected_models:
                default_model = selected_models[0]
            st.session_state["_selected_exploration_model_classification"] = default_model
        
        # Validate that saved value is still valid
        if st.session_state["_selected_exploration_model_classification"] not in selected_models:
            st.session_state["_selected_exploration_model_classification"] = selected_models[0]
        
        selected_model_name = st.selectbox(
            "Choose a model to explore:",
            selected_models,
            index=selected_models.index(st.session_state["_selected_exploration_model_classification"]),
            key="_selected_exploration_model_classification",
            on_change=save_to_state,
            args=("_selected_exploration_model_classification", "selected_exploration_model_classification")
        )
        
        selected_model = models[selected_model_name]
        y_pred_selected = selected_model.predict(X_test)
        
        # Confusion Matrices - Train and Test
        toc_subsubheader(f"🔍 Confusion Matrices - {selected_model_name}")
        
        y_pred_train = selected_model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_selected)
        
        col_train, col_test = st.columns(2)
        
        with col_train:
            fig_train = create_confusion_matrix_train_plot(cm_train)
            st.plotly_chart(fig_train, use_container_width=True)
            
            tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
            total_train = tn_train + fp_train + fn_train + tp_train
            
            st.markdown("#### 📊 Train Set Metrics:")
            st.markdown(f"""
            - **True Negatives:** {tn_train}
            - **False Positives:** {fp_train}
            - **False Negatives:** {fn_train}
            - **True Positives:** {tp_train}
            
            **Accuracy:** {(tp_train+tn_train)/total_train:.1%}
            """)
        
        with col_test:
            fig_test = create_confusion_matrix_plot(cm_test)
            st.plotly_chart(fig_test, use_container_width=True)
            
            tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
            total_test = tn_test + fp_test + fn_test + tp_test
            
            st.markdown("#### 📊 Test Set Metrics:")
            st.markdown(f"""
            - **True Negatives:** {tn_test}
            - **False Positives:** {fp_test}
            - **False Negatives:** {fn_test}
            - **True Positives:** {tp_test}
            
            **Accuracy:** {(tp_test+tn_test)/total_test:.1%}
            """)
        
        # 2D Visualization
        if len(feature_names) >= 2:
            toc_subsubheader(f"📊 2D Feature Visualization - {selected_model_name}")
            
            col1, col2 = st.columns(2)
            with col1:
                # Initialize temp key from permanent key if needed
                if "_clf_2d_x" not in st.session_state:
                    default_x = st.session_state.get("clf_2d_x", feature_names[0])
                    if default_x not in feature_names:
                        default_x = feature_names[0]
                    st.session_state["_clf_2d_x"] = default_x
                
                # Validate that saved value is still valid
                if st.session_state["_clf_2d_x"] not in feature_names:
                    st.session_state["_clf_2d_x"] = feature_names[0]
                
                feature_x = st.selectbox(
                    "Choose X-axis feature:",
                    feature_names,
                    index=feature_names.index(st.session_state["_clf_2d_x"]),
                    key="_clf_2d_x",
                    on_change=save_to_state,
                    args=("_clf_2d_x", "clf_2d_x")
                )
            with col2:
                available_y = [f for f in feature_names if f != feature_x]
                
                # Initialize temp key from permanent key if needed
                if "_clf_2d_y" not in st.session_state:
                    default_y = st.session_state.get("clf_2d_y", available_y[0] if available_y else feature_names[0])
                    if default_y not in available_y:
                        default_y = available_y[0] if available_y else feature_names[0]
                    st.session_state["_clf_2d_y"] = default_y
                
                # Validate that saved value is still valid
                if st.session_state["_clf_2d_y"] not in available_y:
                    st.session_state["_clf_2d_y"] = available_y[0] if available_y else feature_names[0]
                
                feature_y = st.selectbox(
                    "Choose Y-axis feature:",
                    available_y,
                    index=available_y.index(st.session_state["_clf_2d_y"]),
                    key="_clf_2d_y",
                    on_change=save_to_state,
                    args=("_clf_2d_y", "clf_2d_y")
                )
            
            if len(feature_names) == 2:
                st.info(f"🎯 **Perfect!** With exactly 2 features, you can see how {selected_model_name} creates decision boundaries!")
                fig = create_decision_boundary_plot(X_train, y_train, selected_model, feature_x, feature_y, selected_model_name)
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"💡 **Decision Boundary for {selected_model_name}**: The colored regions show where the model predicts each class. Points show actual data.")
            else:
                fig = create_2d_scatter_plot(X_test, y_test, y_pred_selected, feature_x, feature_y, "Classification", "Survived")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"💡 **2D Feature Plot for {selected_model_name}**: Each point is a passenger. Colors show actual vs predicted classes. Look for patterns!")
        
        # Feature Importance
        if hasattr(selected_model, 'feature_importances_'):
            toc_subsubheader(f"🎯 Feature Importance - {selected_model_name}")
            
            fig = create_feature_importance_plot(feature_names, selected_model.feature_importances_, selected_model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Feature Importance for {selected_model_name}**: Shows which passenger characteristics the model considers most important for predicting survival.")
        
        elif hasattr(selected_model, 'coef_'):
            toc_subsubheader(f"🎯 Feature Coefficients - {selected_model_name}")
            
            # Handle both 1D and 2D coefficient arrays
            coef = selected_model.coef_[0] if len(selected_model.coef_.shape) > 1 else selected_model.coef_
            fig = create_feature_coefficients_plot(feature_names, coef, selected_model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Feature Coefficients for {selected_model_name}**: Shows how much each feature influences the model's predictions. Positive values increase survival probability, negative values decrease it.")
        
        # Interactive Prediction
        toc_subsubheader("🔮 Make Your Own Predictions")
        
        st.markdown(f"Try different passenger profiles and see what **{selected_model_name}** predicts!")
        
        col1, col2 = st.columns(2)
        
        # Initialize prediction_inputs before columns to avoid UnboundLocalError
        prediction_inputs = {}
        
        with col1:
            st.subheader("👤 Passenger Profile")
            
            if not feature_names:
                st.warning("⚠️ Please select features in the 'Preprocessing & Exploration' page first to make predictions.")
            else:
                prediction_inputs = render_prediction_inputs(selected_features, "clf")
        
        with col2:
            st.subheader("🎯 Prediction Result")
            
            if selected_features and st.button("🔮 Predict Survival", type="primary", key="clf_predict_btn"):
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
                probability = selected_model.predict_proba(input_array)[0]
                
                display_prediction_result(prediction, probability, y, "Classification")
    
    # Render TOC at the end
    render_toc()


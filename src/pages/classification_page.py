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
    create_comparison_chart, create_decision_boundary_plot,
    create_2d_scatter_plot, create_confusion_matrix_plot,
    create_feature_importance_plot, create_feature_coefficients_plot
)
from ..utils.ui_components import (
    display_metrics_row, render_hyperparameter_controls,
    render_prediction_inputs, display_prediction_result
)


def save_to_state(temp_key, perm_key):
    """Save widget value from temporary key to permanent key."""
    st.session_state[perm_key] = st.session_state[temp_key]


def render_classification_page():
    """Render the classification page."""
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
    
    st.markdown('<h2 class="section-header">🤖 Classification Models Training & Comparison</h2>', unsafe_allow_html=True)
    
    # Model selection in page
    st.markdown("### 📋 Model Selection")
    
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
    
    # Get test size from session state
    test_size = st.session_state.get("test_size", 20)
    
    # Hyperparameters in page
    st.markdown("### ⚙️ Hyperparameter Tuning")
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
        
        st.subheader("📊 Model Performance Comparison")
        
        # Metrics display
        for result in results:
            display_metrics_row(result, "Classification")
        
        # Comparison chart
        fig = create_comparison_chart(results_df, "Classification")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model_name = results_df.loc[results_df['Test Score'].idxmax(), 'Model']
        best_model = models[best_model_name]
        y_pred_best = best_model.predict(X_test)
        
        st.subheader(f"🏆 Best Model: {best_model_name}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_best)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_confusion_matrix_plot(cm)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            
            st.markdown("#### 🔍 Confusion Matrix Explained:")
            st.markdown(f"""
            - **True Negatives (TN):** {tn} - Correctly predicted 'did not survive'
            - **False Positives (FP):** {fp} - Wrongly predicted 'survived' 
            - **False Negatives (FN):** {fn} - Wrongly predicted 'did not survive'
            - **True Positives (TP):** {tp} - Correctly predicted 'survived'
            
            **Overall Accuracy:** {(tp+tn)/total:.1%}
            """)
        
        # 2D Visualization
        if len(feature_names) >= 2:
            st.subheader("📊 2D Feature Visualization")
            
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
                st.info("🎯 **Perfect!** With exactly 2 features, you can see how the classifier creates decision boundaries!")
                fig = create_decision_boundary_plot(X_train, y_train, best_model, feature_x, feature_y, best_model_name)
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Decision Boundary**: The colored regions show where the model predicts each class. Points show actual data.")
            else:
                fig = create_2d_scatter_plot(X_test, y_test, y_pred_best, feature_x, feature_y, "Classification", "Survived")
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **2D Feature Plot**: Each point is a passenger. Colors show actual vs predicted classes. Look for patterns!")
        
        # Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            fig = create_feature_importance_plot(feature_names, best_model.feature_importances_, best_model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Feature Importance** shows which passenger characteristics the {best_model_name} considers most important for predicting survival.")
        
        elif hasattr(best_model, 'coef_'):
            st.subheader("🎯 Feature Coefficients")
            
            # Handle both 1D and 2D coefficient arrays
            coef = best_model.coef_[0] if len(best_model.coef_.shape) > 1 else best_model.coef_
            fig = create_feature_coefficients_plot(feature_names, coef, best_model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Feature Coefficients** show how much each feature influences the {best_model_name}'s predictions. Positive values increase survival probability, negative values decrease it.")
        
        # Interactive Prediction
        st.markdown('<h2 class="section-header">🔮 Make Your Own Predictions</h2>', unsafe_allow_html=True)
        
        st.markdown("Try different passenger profiles and see what the AI predicts!")
        
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
                
                prediction = best_model.predict(input_array)[0]
                probability = best_model.predict_proba(input_array)[0]
                
                display_prediction_result(prediction, probability, y, "Classification")


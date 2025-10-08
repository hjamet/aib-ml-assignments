"""
UI components module for the ML Demo application.
Contains reusable Streamlit UI components.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .model_factory import get_model_descriptions, get_metric_descriptions


# Metrics configuration
METRICS_TO_MAXIMIZE = {'Accuracy', 'Precision', 'Recall', 'F1', 'R²'}
METRICS_TO_MINIMIZE = {'RMSE', 'MAE'}

METRIC_COLUMN_MAPPING = {
    'Classification': {
        'Accuracy': 'Accuracy',
        'Precision': 'Precision', 
        'Recall': 'Recall',
        'F1': 'F1'
    },
    'Regression': {
        'R²': 'R²',
        'RMSE': 'RMSE',
        'MAE': 'MAE'
    }
}


def display_dataset_overview(df):
    """Display the dataset overview metrics."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        st.metric("Survival Rate", f"{df['survived'].mean():.1%}")
    with col3:
        st.metric("Features", df.shape[1])
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())


def display_preprocessing_results(before_count, after_count, n_features, feature_list):
    """Display preprocessing results."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Samples", before_count)
    with col2:
        st.metric("Final Samples", after_count)
    with col3:
        st.metric("Features Used", n_features)
    
    if before_count != after_count:
        st.warning(f"⚠️ Removed {before_count - after_count} rows with missing values")


def display_metrics_table(results_df, problem_type, selected_metrics):
    """
    Display metrics in two side-by-side tables (Train Set and Test Set).
    
    Args:
        results_df (pd.DataFrame): DataFrame containing model results
        problem_type (str): Either "Classification" or "Regression"
        selected_metrics (list): List of selected metrics to display
    """
    if results_df.empty or not selected_metrics:
        st.warning("⚠️ No models or metrics selected to display.")
        return
    
    # Create two columns for Train and Test tables
    col_train, col_test = st.columns(2)
    
    # Prepare data for Train Set table
    train_data = {'Model': results_df['Model'].tolist()}
    test_data = {'Model': results_df['Model'].tolist()}
    
    # Map selected metrics to column names
    metric_mapping = METRIC_COLUMN_MAPPING[problem_type]
    
    for metric in selected_metrics:
        column_name = metric_mapping.get(metric, metric)
        train_col = f'Train {column_name}'
        test_col = f'Test {column_name}'
        
        if train_col in results_df.columns:
            train_data[metric] = results_df[train_col].tolist()
        if test_col in results_df.columns:
            test_data[metric] = results_df[test_col].tolist()
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Function to format values
    def format_value(val, metric):
        """Format metric value based on its type."""
        if pd.isna(val):
            return "N/A"
        
        # Values between 0 and 1 (except R²) as percentage
        if metric in {'Accuracy', 'Precision', 'Recall', 'F1'}:
            return f"{val:.1%}"
        elif metric == 'R²':
            return f"{val:.3f}"
        else:  # RMSE, MAE
            return f"{val:.2f}"
    
    # Function to add trophy emoji to best scores
    def add_trophy_to_best(df, metric):
        """Add trophy emoji to the best score for a metric."""
        if metric not in df.columns or len(df) == 0:
            return df
        
        values = df[metric].copy()
        
        # Determine if we're looking for max or min
        if metric in METRICS_TO_MAXIMIZE:
            best_idx = values.idxmax()
        elif metric in METRICS_TO_MINIMIZE:
            best_idx = values.idxmin()
        else:
            return df
        
        # Format all values
        formatted_values = [format_value(val, metric) for val in values]
        
        # Add trophy to best score
        if not pd.isna(values.iloc[best_idx]):
            formatted_values[best_idx] = f"{formatted_values[best_idx]} 🏆"
        
        df[metric] = formatted_values
        return df
    
    # Format and add trophies to metrics
    for metric in selected_metrics:
        if metric in train_df.columns:
            train_df = add_trophy_to_best(train_df, metric)
        if metric in test_df.columns:
            test_df = add_trophy_to_best(test_df, metric)
    
    # Display tables
    with col_train:
        st.markdown("**Train Set Performance**")
        st.dataframe(train_df, use_container_width=True, hide_index=True)
    
    with col_test:
        st.markdown("**Test Set Performance**")
        st.dataframe(test_df, use_container_width=True, hide_index=True)


def display_model_metrics_columns(result, problem_type):
    """Display metrics for a single model in two columns: Train Set and Test Set."""
    st.markdown(f"#### {result['Model']}")
    
    col_train, col_test = st.columns(2)
    
    if problem_type == "Classification":
        with col_train:
            st.markdown("**Train Set**")
            st.metric("Accuracy", f"{result['Train Accuracy']:.1%}")
            st.metric("Precision", f"{result['Train Precision']:.1%}")
            st.metric("Recall", f"{result['Train Recall']:.1%}")
            st.metric("F1-Score", f"{result['Train F1']:.1%}")
        
        with col_test:
            st.markdown("**Test Set**")
            st.metric("Accuracy", f"{result['Test Accuracy']:.1%}")
            st.metric("Precision", f"{result['Test Precision']:.1%}")
            st.metric("Recall", f"{result['Test Recall']:.1%}")
            st.metric("F1-Score", f"{result['Test F1']:.1%}")
    
    else:  # Regression
        with col_train:
            st.markdown("**Train Set**")
            st.metric("R² Score", f"{result['Train R²']:.3f}")
            st.metric("RMSE", f"{result['Train RMSE']:.2f}")
            st.metric("MAE", f"{result['Train MAE']:.2f}")
        
        with col_test:
            st.markdown("**Test Set**")
            st.metric("R² Score", f"{result['Test R²']:.3f}")
            st.metric("RMSE", f"{result['Test RMSE']:.2f}")
            st.metric("MAE", f"{result['Test MAE']:.2f}")
    
    st.markdown("---")


def render_hyperparameter_controls(model_name, problem_type, key_prefix):
    """
    Render hyperparameter controls for a specific model.
    
    Args:
        model_name (str): Name of the model
        problem_type (str): Either "Classification" or "Regression"
        key_prefix (str): Prefix for widget keys
        
    Returns:
        dict: Dictionary of hyperparameters
    """
    hyperparams = {}
    
    if model_name == "Logistic Regression":
        hyperparams = {
            'C': st.slider(f"Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"{key_prefix}_lr_c"),
            'max_iter': st.slider(f"Max Iterations", 100, 2000, 1000, 100, key=f"{key_prefix}_lr_iter"),
            'solver': st.selectbox(f"Solver", ['liblinear', 'lbfgs', 'saga'], index=1, key=f"{key_prefix}_lr_solver")
        }
        
    elif model_name == "Random Forest":
        hyperparams = {
            'n_estimators': st.slider(f"Number of Trees", 10, 500, 100, 10, key=f"{key_prefix}_rf_trees"),
            'max_depth': st.slider(f"Max Depth", 3, 20, 10, 1, key=f"{key_prefix}_rf_depth"),
            'min_samples_split': st.slider(f"Min Samples Split", 2, 20, 2, 1, key=f"{key_prefix}_rf_split"),
            'min_samples_leaf': st.slider(f"Min Samples Leaf", 1, 10, 1, 1, key=f"{key_prefix}_rf_leaf")
        }
        
    elif model_name == "Support Vector Machine":
        hyperparams = {
            'C': st.slider(f"Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"{key_prefix}_svm_c"),
            'kernel': st.selectbox(f"Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], key=f"{key_prefix}_svm_kernel"),
            'gamma': st.selectbox(f"Gamma", ['scale', 'auto'], key=f"{key_prefix}_svm_gamma")
        }
        
    elif model_name == "Support Vector Regression":
        hyperparams = {
            'C': st.slider(f"Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"{key_prefix}_svr_c"),
            'kernel': st.selectbox(f"Kernel", ['rbf', 'linear', 'poly'], key=f"{key_prefix}_svr_kernel"),
            'epsilon': st.slider(f"Epsilon", 0.01, 1.0, 0.1, 0.01, key=f"{key_prefix}_svr_epsilon")
        }
        
    elif model_name == "Decision Tree":
        hyperparams = {
            'max_depth': st.slider(f"Max Depth", 3, 20, 10, 1, key=f"{key_prefix}_dt_depth"),
            'min_samples_split': st.slider(f"Min Samples Split", 2, 20, 2, 1, key=f"{key_prefix}_dt_split"),
            'min_samples_leaf': st.slider(f"Min Samples Leaf", 1, 10, 1, 1, key=f"{key_prefix}_dt_leaf")
        }
        if problem_type == "Classification":
            hyperparams['criterion'] = st.selectbox(f"Criterion", ['gini', 'entropy'], key=f"{key_prefix}_dt_criterion")
            
    elif model_name == "K-Nearest Neighbors":
        hyperparams = {
            'n_neighbors': st.slider(f"Number of Neighbors", 1, 20, 5, 1, key=f"{key_prefix}_knn_n"),
            'weights': st.selectbox(f"Weights", ['uniform', 'distance'], key=f"{key_prefix}_knn_weights"),
            'metric': st.selectbox(f"Distance Metric", ['euclidean', 'manhattan', 'minkowski'], key=f"{key_prefix}_knn_metric")
        }
        
    elif model_name == "Gradient Boosting":
        hyperparams = {
            'n_estimators': st.slider(f"Number of Estimators", 50, 300, 100, 10, key=f"{key_prefix}_gb_trees"),
            'learning_rate': st.slider(f"Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"{key_prefix}_gb_lr"),
            'max_depth': st.slider(f"Max Depth", 3, 10, 6, 1, key=f"{key_prefix}_gb_depth")
        }
        
    elif model_name == "AdaBoost":
        hyperparams = {
            'n_estimators': st.slider(f"Number of Estimators", 10, 200, 50, 10, key=f"{key_prefix}_ada_trees"),
            'learning_rate': st.slider(f"Learning Rate", 0.1, 2.0, 1.0, 0.1, key=f"{key_prefix}_ada_lr"),
            'algorithm': st.selectbox(f"Algorithm", ['SAMME', 'SAMME.R'], key=f"{key_prefix}_ada_algo")
        }
        
    elif model_name == "Naive Bayes":
        hyperparams = {
            'var_smoothing': st.slider(f"Smoothing", 1e-10, 1e-5, 1e-9, 1e-10, key=f"{key_prefix}_nb_smooth", format="%.2e")
        }
        
    elif model_name == "Neural Network":
        layer_sizes = st.multiselect(f"Hidden Layer Sizes", [50, 100, 200, 300], default=[100], key=f"{key_prefix}_nn_layers")
        hyperparams = {
            'hidden_layer_sizes': tuple(layer_sizes) if layer_sizes else (100,),
            'learning_rate_init': st.slider(f"Learning Rate", 0.001, 0.1, 0.001, 0.001, key=f"{key_prefix}_nn_lr"),
            'max_iter': st.slider(f"Max Iterations", 100, 1000, 200, 50, key=f"{key_prefix}_nn_iter"),
            'activation': st.selectbox(f"Activation", ['relu', 'tanh', 'logistic'], key=f"{key_prefix}_nn_activation")
        }
        
    elif model_name == "Ridge Classifier" or model_name == "Ridge Regression":
        hyperparams = {
            'alpha': st.slider(f"Alpha (Regularization)", 0.1, 10.0, 1.0, 0.1, key=f"{key_prefix}_ridge_alpha")
        }
        
    elif model_name == "Linear Regression":
        hyperparams = {
            'fit_intercept': st.checkbox(f"Fit Intercept", value=True, key=f"{key_prefix}_linreg_intercept")
        }
        
    elif model_name == "Lasso Regression":
        hyperparams = {
            'alpha': st.slider(f"Alpha", 0.01, 2.0, 1.0, 0.01, key=f"{key_prefix}_lasso_alpha"),
            'max_iter': st.slider(f"Max Iterations", 100, 2000, 1000, 100, key=f"{key_prefix}_lasso_iter")
        }
    
    return hyperparams


def render_prediction_inputs(selected_features, key_prefix):
    """
    Render input controls for making predictions.
    
    Args:
        selected_features (list): List of selected features
        key_prefix (str): Prefix for widget keys
        
    Returns:
        dict: Dictionary of prediction inputs mapped to feature names
    """
    prediction_inputs = {}
    
    if 'Age' in selected_features:
        prediction_inputs['age'] = st.slider("Age", 0, 80, 30, key=f"{key_prefix}_pred_age")
    if 'Sex' in selected_features:
        sex_input = st.selectbox("Gender", ["Female", "Male"], key=f"{key_prefix}_pred_sex")
        prediction_inputs['sex_encoded'] = 1 if sex_input == "Male" else 0
    if 'Passenger Class' in selected_features:
        prediction_inputs['pclass'] = st.selectbox("Passenger Class", [1, 2, 3], index=1, key=f"{key_prefix}_pred_pclass")
    if 'Fare' in selected_features:
        prediction_inputs['fare'] = st.slider("Fare ($)", 0, 500, 50, key=f"{key_prefix}_pred_fare")
    if 'Siblings/Spouses' in selected_features:
        prediction_inputs['sibsp'] = st.slider("Siblings/Spouses Aboard", 0, 8, 0, key=f"{key_prefix}_pred_sibsp")
    if 'Parents/Children' in selected_features:
        prediction_inputs['parch'] = st.slider("Parents/Children Aboard", 0, 6, 0, key=f"{key_prefix}_pred_parch")
    if 'Port of Embarkation' in selected_features:
        port_input = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"], key=f"{key_prefix}_pred_port")
        port_mapping = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
        prediction_inputs['embarked_encoded'] = port_mapping[port_input]
    
    return prediction_inputs


def display_prediction_result(prediction, probability, y, problem_type):
    """
    Display prediction results.
    
    Args:
        prediction: The predicted value
        probability: Prediction probabilities (for classification)
        y: The target series (for context)
        problem_type (str): Either "Classification" or "Regression"
    """
    if problem_type == "Classification":
        if prediction == 1:
            st.success(f"🎉 **SURVIVAL PREDICTED!**")
            st.success(f"Confidence: {probability[1]:.1%}")
        else:
            st.error(f"💔 **Did not survive**")
            st.error(f"Confidence: {probability[0]:.1%}")
        
        # Show probability breakdown
        import pandas as pd
        prob_df = pd.DataFrame({
            'Outcome': ['Did not survive', 'Survived'],
            'Probability': probability
        })
        
        fig = px.bar(prob_df, x='Outcome', y='Probability', 
                   title='Prediction Confidence',
                   color='Probability',
                   color_continuous_scale='RdYlGn')
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:  # Regression
        st.success(f"🎯 **Predicted Value: {prediction:.2f}**")
        
        # Show prediction context
        y_min, y_max = y.min(), y.max()
        y_mean = y.mean()
        
        st.info(f"""
        **📊 Prediction Context:**
        - **Predicted Value:** {prediction:.2f}
        - **Dataset Range:** {y_min:.2f} to {y_max:.2f}  
        - **Dataset Average:** {y_mean:.2f}
        - **Prediction vs Average:** {((prediction - y_mean) / y_mean * 100):+.1f}%
        """)
        
        # Visual prediction context
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y, name='Dataset Distribution', opacity=0.7))
        fig.add_vline(x=prediction, line_dash="dash", line_color="red", 
                     annotation_text=f"Prediction: {prediction:.2f}")
        fig.add_vline(x=y_mean, line_dash="dot", line_color="blue",
                     annotation_text=f"Average: {y_mean:.2f}")
        fig.update_layout(title="Your Prediction vs Dataset Distribution")
        st.plotly_chart(fig, use_container_width=True)


def inject_card_styles():
    """Inject CSS styles for modern card design with hover animations."""
    st.markdown("""
        <style>
        .info-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            border: 1px solid #e9ecef;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        }
        
        .card-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
        }
        
        .card-emoji {
            font-size: 1.5em;
            margin-right: 10px;
        }
        
        .card-description {
            color: #495057;
            line-height: 1.6;
            margin-bottom: 12px;
            font-size: 0.95em;
        }
        
        .card-section {
            margin-top: 10px;
        }
        
        .card-section-title {
            font-weight: 600;
            color: #495057;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .card-list {
            list-style: none;
            padding-left: 0;
            margin: 5px 0;
        }
        
        .card-list li {
            padding: 3px 0;
            color: #6c757d;
            font-size: 0.85em;
        }
        
        .card-list li:before {
            content: "✓ ";
            color: #28a745;
            font-weight: bold;
            margin-right: 5px;
        }
        
        .card-formula {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            border-left: 3px solid #007bff;
            margin: 8px 0;
            font-family: 'Courier New', monospace;
        }
        
        .weakness-list li:before {
            content: "⚠ ";
            color: #ffc107;
        }
        </style>
    """, unsafe_allow_html=True)


def render_model_info_cards(selected_models, problem_type):
    """
    Render information cards for selected models.
    
    Args:
        selected_models (list): List of selected model names
        problem_type (str): Either "Classification" or "Regression"
    """
    if not selected_models:
        return
    
    model_descriptions = get_model_descriptions(problem_type)
    
    # Display in a grid with 4 columns
    num_cols = min(4, len(selected_models))
    cols = st.columns(num_cols)
    
    for idx, model_name in enumerate(selected_models):
        col_idx = idx % num_cols
        model_info = model_descriptions.get(model_name, {})
        
        with cols[col_idx]:
            emoji = model_info.get("emoji", "🤖")
            description = model_info.get("description", "")
            strengths = model_info.get("strengths", [])
            best_for = model_info.get("best_for", "")
            
            # Simple HTML card with just title and description
            st.markdown(f"""
            <div class="info-card">
                <div class="card-title">
                    <span class="card-emoji">{emoji}</span>
                    <span>{model_name}</span>
                </div>
                <div class="card-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Use native Streamlit components for additional info
            if strengths:
                st.markdown("**Strengths:**")
                for strength in strengths:
                    st.markdown(f"✓ {strength}")
            
            if best_for:
                st.markdown("**Best for:**")
                st.markdown(f"*{best_for}*")


def render_metric_info_cards(selected_metrics, problem_type):
    """
    Render information cards for selected metrics.
    
    Args:
        selected_metrics (list): List of selected metric names
        problem_type (str): Either "Classification" or "Regression"
    """
    if not selected_metrics:
        return
    
    metric_descriptions = get_metric_descriptions(problem_type)
    
    # Display in a grid with 4 columns
    num_cols = min(4, len(selected_metrics))
    cols = st.columns(num_cols)
    
    for idx, metric_name in enumerate(selected_metrics):
        col_idx = idx % num_cols
        metric_info = metric_descriptions.get(metric_name, {})
        
        with cols[col_idx]:
            emoji = metric_info.get("emoji", "📊")
            description = metric_info.get("description", "")
            formula = metric_info.get("formula", "")
            usage = metric_info.get("usage", "")
            strengths = metric_info.get("strengths", [])
            weaknesses = metric_info.get("weaknesses", [])
            
            # Build content HTML to put everything inside the card
            content_html = f'<div class="card-description">{description}</div>'
            
            if usage:
                content_html += f'<div class="card-section"><div class="card-section-title">When to use:</div><div style="color: #6c757d; font-size: 0.85em;">{usage}</div></div>'
            
            if strengths:
                content_html += '<div class="card-section"><div class="card-section-title">Strengths:</div><ul class="card-list">'
                for strength in strengths:
                    content_html += f'<li>{strength}</li>'
                content_html += '</ul></div>'
            
            if weaknesses:
                content_html += '<div class="card-section"><div class="card-section-title">Weaknesses:</div><ul class="card-list weakness-list">'
                for weakness in weaknesses:
                    content_html += f'<li>{weakness}</li>'
                content_html += '</ul></div>'
            
            # Display complete card with all content inside
            st.markdown(f"""
            <div class="info-card">
                <div class="card-title">
                    <span class="card-emoji">{emoji}</span>
                    <span>{metric_name}</span>
                </div>
                {content_html}
            </div>
            """, unsafe_allow_html=True)
            
            # Display formula below the card (as requested, formula should be included in card)
            if formula:
                st.markdown(f"""
                <div class="info-card" style="margin-top: -10px; padding-top: 10px;">
                    <div class="card-section">
                        <div class="card-section-title">Formula:</div>
                        <div style="text-align: center; padding: 10px;">
                            ${formula}$
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


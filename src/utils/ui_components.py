"""
UI components module for the ML Demo application.
Contains reusable Streamlit UI components.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


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
    
    st.success(f"✅ Data preprocessing completed! Using {n_features} features: {', '.join(feature_list)}")


def display_metrics_row(result, problem_type):
    """Display a row of metrics for a single model."""
    if problem_type == "Classification":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{result['Model']} - Accuracy", f"{result['Test Score']:.1%}")
        with col2:
            st.metric("Precision", f"{result['Precision']:.1%}")
        with col3:
            st.metric("Recall", f"{result['Recall']:.1%}")
        with col4:
            overfitting_status = "⚠️" if result['Overfitting'] > 0.05 else "✅"
            st.metric("Overfitting", f"{overfitting_status} {result['Overfitting']:.1%}")
    else:  # Regression
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{result['Model']} - R²", f"{result['Test Score']:.3f}")
        with col2:
            st.metric("RMSE", f"{result['RMSE']:.2f}")
        with col3:
            st.metric("MAE", f"{result['MAE']:.2f}")
        with col4:
            overfitting_status = "⚠️" if result['Overfitting'] > 0.1 else "✅"
            st.metric("Overfitting", f"{overfitting_status} {result['Overfitting']:.3f}")


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
        st.plotly_chart(fig, width='stretch')
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
        st.plotly_chart(fig, width='stretch')


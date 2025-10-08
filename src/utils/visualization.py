"""
Visualization module for the ML Demo application.
Contains all plotting and chart creation functions.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_decision_boundary_plot(X_train, y_train, model, feature_x, feature_y, model_name):
    """Create a decision boundary plot for 2D classification"""
    # Get the two features
    X_2d = X_train[[feature_x, feature_y]]
    
    # Create a mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X_2d[feature_x].min() - 1, X_2d[feature_x].max() + 1
    y_min, y_max = X_2d[feature_y].min() - 1, X_2d[feature_y].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    fig = go.Figure()
    
    # Add decision boundary
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale='RdYlBu',
        opacity=0.6,
        showscale=False,
        contours=dict(start=0, end=1, size=1),
        name="Decision Boundary"
    ))
    
    # Add training points
    for class_val in y_train.unique():
        mask = y_train == class_val
        fig.add_trace(go.Scatter(
            x=X_2d.loc[mask, feature_x],
            y=X_2d.loc[mask, feature_y],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(size=8, line=dict(width=1, color='black'))
        ))
    
    fig.update_layout(
        title=f'Decision Boundary: {model_name}',
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        width=800,
        height=600
    )
    
    return fig


def create_regression_surface_plot(X_train, y_train, model, feature_x, feature_y, model_name, target_name):
    """Create a 3D surface plot for 2D regression"""
    # Get the two features
    X_2d = X_train[[feature_x, feature_y]]
    
    # Create a mesh grid
    x_min, x_max = X_2d[feature_x].min(), X_2d[feature_x].max()
    y_min, y_max = X_2d[feature_y].min(), X_2d[feature_y].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50)
    )
    
    # Make predictions on the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create the 3D surface plot
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=Z,
        colorscale='Viridis',
        opacity=0.8,
        name="Prediction Surface"
    ))
    
    # Add training points
    fig.add_trace(go.Scatter3d(
        x=X_2d[feature_x],
        y=X_2d[feature_y],
        z=y_train,
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Training Data'
    ))
    
    fig.update_layout(
        title=f'Regression Surface: {model_name}',
        scene=dict(
            xaxis_title=feature_x,
            yaxis_title=feature_y,
            zaxis_title=target_name
        ),
        width=800,
        height=600
    )
    
    return fig


def create_2d_scatter_plot(X_test, y_test, y_pred, feature_x, feature_y, problem_type, target_name):
    """Create a 2D scatter plot showing predictions vs actual"""
    fig = go.Figure()
    
    if problem_type == "Classification":
        # Show actual vs predicted classes
        for class_val in y_test.unique():
            mask = y_test == class_val
            fig.add_trace(go.Scatter(
                x=X_test.loc[mask, feature_x],
                y=X_test.loc[mask, feature_y],
                mode='markers',
                name=f'Actual: {class_val}',
                marker=dict(
                    size=8,
                    symbol='circle',
                    line=dict(width=2, color='black')
                )
            ))
        
        # Add predictions as markers
        correct_pred = y_test == y_pred
        fig.add_trace(go.Scatter(
            x=X_test.loc[~correct_pred, feature_x],
            y=X_test.loc[~correct_pred, feature_y],
            mode='markers',
            name='Wrong Predictions',
            marker=dict(size=12, symbol='x', color='red', line=dict(width=2))
        ))
        
    else:  # Regression
        # Color by actual values, size by prediction error
        error = np.abs(y_test - y_pred)
        fig.add_trace(go.Scatter(
            x=X_test[feature_x],
            y=X_test[feature_y],
            mode='markers',
            marker=dict(
                size=10 + error * 5,  # Size based on error
                color=y_test,  # Color based on actual values
                colorscale='Viridis',
                colorbar=dict(title=target_name),
                line=dict(width=1, color='black')
            ),
            text=[f'Actual: {actual:.2f}<br>Predicted: {pred:.2f}<br>Error: {err:.2f}' 
                  for actual, pred, err in zip(y_test, y_pred, error)],
            hovertemplate='%{text}<extra></extra>',
            name='Test Data'
        ))
    
    fig.update_layout(
        title=f'2D Feature Plot: {feature_x} vs {feature_y}',
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        width=800,
        height=600
    )
    
    return fig


def create_train_metrics_chart(results_df, problem_type, selected_metrics=None):
    """Create a bar chart showing training set metrics for all models"""
    models = results_df['Model'].tolist()
    fig = go.Figure()
    
    if problem_type == "Classification":
        # Metric configuration for classification
        metric_config = {
            'Accuracy': {'column': 'Train Accuracy', 'color': 'lightblue'},
            'Precision': {'column': 'Train Precision', 'color': 'lightgreen'},
            'Recall': {'column': 'Train Recall', 'color': 'lightsalmon'},
            'F1': {'column': 'Train F1', 'color': 'lightcoral'}
        }
        
        # Filter metrics if selected_metrics is provided
        if selected_metrics:
            metrics_to_show = [m for m in selected_metrics if m in metric_config]
        else:
            metrics_to_show = list(metric_config.keys())
        
        # Add traces for selected metrics
        for metric in metrics_to_show:
            config = metric_config[metric]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=results_df[config['column']],
                marker_color=config['color']
            ))
        
        fig.update_layout(
            title='Training Set Performance',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 1])
        )
    else:  # Regression
        # Metric configuration for regression - all metrics on same axis
        metric_config = {
            'R²': {'column': 'Train R²', 'color': 'lightblue'},
            'RMSE': {'column': 'Train RMSE', 'color': 'lightcoral'},
            'MAE': {'column': 'Train MAE', 'color': 'lightgreen'}
        }
        
        # Filter metrics if selected_metrics is provided
        if selected_metrics:
            metrics_to_show = [m for m in selected_metrics if m in metric_config]
        else:
            metrics_to_show = list(metric_config.keys())
        
        # Add traces for selected metrics
        for metric in metrics_to_show:
            config = metric_config[metric]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=results_df[config['column']],
                marker_color=config['color']
            ))
        
        # Configure layout
        fig.update_layout(
            title='Training Set Performance',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
    
    return fig


def create_test_metrics_chart(results_df, problem_type, selected_metrics=None):
    """Create a bar chart showing test set metrics for all models"""
    models = results_df['Model'].tolist()
    fig = go.Figure()
    
    if problem_type == "Classification":
        # Metric configuration for classification
        metric_config = {
            'Accuracy': {'column': 'Test Accuracy', 'color': 'lightblue'},
            'Precision': {'column': 'Test Precision', 'color': 'lightgreen'},
            'Recall': {'column': 'Test Recall', 'color': 'lightsalmon'},
            'F1': {'column': 'Test F1', 'color': 'lightcoral'}
        }
        
        # Filter metrics if selected_metrics is provided
        if selected_metrics:
            metrics_to_show = [m for m in selected_metrics if m in metric_config]
        else:
            metrics_to_show = list(metric_config.keys())
        
        # Add traces for selected metrics
        for metric in metrics_to_show:
            config = metric_config[metric]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=results_df[config['column']],
                marker_color=config['color']
            ))
        
        fig.update_layout(
            title='Test Set Performance',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 1])
        )
    else:  # Regression
        # Metric configuration for regression - all metrics on same axis
        metric_config = {
            'R²': {'column': 'Test R²', 'color': 'lightblue'},
            'RMSE': {'column': 'Test RMSE', 'color': 'lightcoral'},
            'MAE': {'column': 'Test MAE', 'color': 'lightgreen'}
        }
        
        # Filter metrics if selected_metrics is provided
        if selected_metrics:
            metrics_to_show = [m for m in selected_metrics if m in metric_config]
        else:
            metrics_to_show = list(metric_config.keys())
        
        # Add traces for selected metrics
        for metric in metrics_to_show:
            config = metric_config[metric]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=results_df[config['column']],
                marker_color=config['color']
            ))
        
        # Configure layout
        fig.update_layout(
            title='Test Set Performance',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
    
    return fig


def create_confusion_matrix_plot(cm, title="Confusion Matrix - Test Set"):
    """Create a heatmap for the confusion matrix"""
    fig = px.imshow(cm, 
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Did not survive', 'Survived'],
                   y=['Did not survive', 'Survived'],
                   color_continuous_scale='Blues',
                   title=title)
    fig.update_traces(text=cm, texttemplate="%{text}")
    return fig


def create_confusion_matrix_train_plot(cm):
    """Create a heatmap for the confusion matrix on training set"""
    return create_confusion_matrix_plot(cm, title="Confusion Matrix - Train Set")


def create_feature_importance_plot(feature_names, importances, model_name):
    """Create a bar plot for feature importances"""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
               title=f'Which Features Matter Most? ({model_name})',
               color_discrete_sequence=['lightgreen'])
    return fig


def create_feature_coefficients_plot(feature_names, coefficients, model_name):
    """Create a bar plot for feature coefficients"""
    feature_coef = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=True)
    
    fig = px.bar(feature_coef, x='coefficient', y='feature', orientation='h',
               title=f'Feature Coefficients ({model_name})',
               color='coefficient', color_continuous_scale='RdBu_r')
    return fig


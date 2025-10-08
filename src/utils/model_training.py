"""
Model training module for the ML Demo application.
Handles model training and evaluation.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error
)

from .model_factory import create_model


def evaluate_classification_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a classification model.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_score = accuracy_score(y_train, train_pred)
    test_score = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
    
    return {
        'Training Score': train_score,
        'Test Score': test_score,
        'Precision': precision,
        'Recall': recall,
        'Overfitting': train_score - test_score,
        'Metric': 'Accuracy'
    }


def evaluate_regression_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a regression model.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_score = r2_score(y_train, train_pred)
    test_score = r2_score(y_test, test_pred)
    mse = mean_squared_error(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mse)
    
    return {
        'Training Score': train_score,
        'Test Score': test_score,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Overfitting': train_score - test_score,
        'Metric': 'R² Score'
    }


def train_models(X, y, selected_models, hyperparams, test_size, problem_type):
    """
    Train multiple models and evaluate them.
    
    Args:
        X: Features
        y: Target
        selected_models (list): List of model names to train
        hyperparams (dict): Dictionary of hyperparameters for each model
        test_size (float): Test set size percentage
        problem_type (str): Either "Classification" or "Regression"
        
    Returns:
        tuple: (models_dict, results_list, X_train, X_test, y_train, y_test)
    """
    # Split data
    if problem_type == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
    
    models = {}
    results = []
    
    for model_name in selected_models:
        # Create model
        model = create_model(model_name, problem_type, hyperparams)
        
        # Train model
        model.fit(X_train, y_train)
        models[model_name] = model
        
        # Evaluate model
        if problem_type == "Classification":
            metrics = evaluate_classification_model(model, X_train, y_train, X_test, y_test)
        else:
            metrics = evaluate_regression_model(model, X_train, y_train, X_test, y_test)
        
        metrics['Model'] = model_name
        results.append(metrics)
    
    return models, results, X_train, X_test, y_train, y_test


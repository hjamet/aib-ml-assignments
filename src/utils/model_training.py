"""
Model training module for the ML Demo application.
Handles model training and evaluation.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
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
        dict: Dictionary containing evaluation metrics for train and test sets
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Train metrics
    train_accuracy = accuracy_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred, average='weighted', zero_division=0)
    train_recall = recall_score(y_train, train_pred, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
    
    return {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1': test_f1,
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
        dict: Dictionary containing evaluation metrics for train and test sets
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Train metrics
    train_r2 = r2_score(y_train, train_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, train_pred)
    
    # Test metrics
    test_r2 = r2_score(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    return {
        'Train R²': train_r2,
        'Train RMSE': train_rmse,
        'Train MAE': train_mae,
        'Test R²': test_r2,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
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
        # Check if stratified split is possible
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        can_stratify = min_class_count >= 2
        
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
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


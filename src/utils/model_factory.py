"""
Model factory module for the ML Demo application.
Handles model instantiation and configuration.
"""

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor


def get_available_models(problem_type):
    """
    Get available models for the specified problem type.
    
    Args:
        problem_type (str): Either "Classification" or "Regression"
        
    Returns:
        dict: Dictionary mapping model names to descriptions
    """
    if problem_type == "Classification":
        return {
            "Logistic Regression": "📊 Simple linear classifier",
            "Random Forest": "🌳 Ensemble of decision trees", 
            "Support Vector Machine": "🎯 Finds optimal decision boundary",
            "Decision Tree": "🌲 Tree-based decision making",
            "K-Nearest Neighbors": "👥 Classification by similarity",
            "Gradient Boosting": "🚀 Advanced ensemble method",
            "AdaBoost": "⚡ Adaptive boosting ensemble",
            "Naive Bayes": "📈 Probabilistic classifier",
            "Neural Network": "🧠 Multi-layer perceptron",
            "Ridge Classifier": "📏 Regularized linear classifier"
        }
    else:  # Regression
        return {
            "Linear Regression": "📈 Simple linear regression",
            "Random Forest": "🌳 Ensemble of regression trees",
            "Support Vector Regression": "🎯 SVM for continuous values",
            "Decision Tree": "🌲 Tree-based regression",
            "K-Nearest Neighbors": "👥 Regression by similarity",
            "Gradient Boosting": "🚀 Advanced ensemble regression",
            "Ridge Regression": "📏 Regularized linear regression",
            "Lasso Regression": "🎯 Feature-selecting linear regression",
            "Neural Network": "🧠 Multi-layer perceptron regression"
        }


def get_model_descriptions(problem_type):
    """
    Get detailed model descriptions for UI display.
    
    Args:
        problem_type (str): Either "Classification" or "Regression"
        
    Returns:
        dict: Dictionary mapping model names to description dictionaries
    """
    if problem_type == "Classification":
        return {
            "Logistic Regression": {
                "emoji": "📊", "description": "A linear model that uses the logistic function to model probabilities.",
                "strengths": ["Fast training", "Interpretable", "Good baseline"], "best_for": "Linear relationships"
            },
            "Random Forest": {
                "emoji": "🌳", "description": "An ensemble method that combines multiple decision trees for classification.",
                "strengths": ["Handles non-linear data", "Robust to outliers", "Feature importance"], "best_for": "Complex patterns"
            },
            "Support Vector Machine": {
                "emoji": "🎯", "description": "Finds optimal hyperplane to separate classes with maximum margin.",
                "strengths": ["Effective in high dimensions", "Memory efficient"], "best_for": "High-dimensional data"
            },
            "Decision Tree": {
                "emoji": "🌲", "description": "Tree-like model that makes decisions by splitting data based on features.",
                "strengths": ["Highly interpretable", "No preprocessing needed"], "best_for": "Interpretability"
            },
            "K-Nearest Neighbors": {
                "emoji": "👥", "description": "Classifies based on majority class of k nearest neighbors.",
                "strengths": ["Simple concept", "No training period"], "best_for": "Small datasets"
            },
            "Gradient Boosting": {
                "emoji": "🚀", "description": "Builds models sequentially, correcting errors of previous ones.",
                "strengths": ["High accuracy", "Handles missing values"], "best_for": "High accuracy requirements"
            },
            "AdaBoost": {
                "emoji": "⚡", "description": "Adaptive boosting focusing on misclassified examples.",
                "strengths": ["Good performance", "Reduces bias"], "best_for": "Binary classification"
            },
            "Naive Bayes": {
                "emoji": "📈", "description": "Probabilistic classifier based on Bayes' theorem.",
                "strengths": ["Fast training", "Works with small datasets"], "best_for": "Text classification"
            },
            "Neural Network": {
                "emoji": "🧠", "description": "Multi-layer perceptron for classification tasks.",
                "strengths": ["Learns complex patterns", "Flexible"], "best_for": "Complex patterns, large datasets"
            },
            "Ridge Classifier": {
                "emoji": "📏", "description": "Linear classifier with L2 regularization.",
                "strengths": ["Handles multicollinearity", "Prevents overfitting"], "best_for": "Linear problems"
            }
        }
    else:  # Regression
        return {
            "Linear Regression": {
                "emoji": "📈", "description": "Simple linear regression modeling relationships between features and target.",
                "strengths": ["Fast training", "Interpretable", "Good baseline"], "best_for": "Linear relationships"
            },
            "Random Forest": {
                "emoji": "🌳", "description": "An ensemble method that combines multiple decision trees for regression.",
                "strengths": ["Handles non-linear data", "Robust to outliers", "Feature importance"], "best_for": "Complex patterns"
            },
            "Support Vector Regression": {
                "emoji": "🎯", "description": "Uses support vector machines for continuous value prediction.",
                "strengths": ["Effective in high dimensions", "Memory efficient"], "best_for": "Non-linear regression"
            },
            "Decision Tree": {
                "emoji": "🌲", "description": "Tree-like model that predicts continuous values by splitting data.",
                "strengths": ["Highly interpretable", "No preprocessing needed"], "best_for": "Interpretability"
            },
            "K-Nearest Neighbors": {
                "emoji": "👥", "description": "Predicts values based on average of k nearest neighbors.",
                "strengths": ["Simple concept", "No training period"], "best_for": "Small datasets"
            },
            "Gradient Boosting": {
                "emoji": "🚀", "description": "Builds regression models sequentially, correcting errors of previous ones.",
                "strengths": ["High accuracy", "Handles missing values"], "best_for": "High accuracy requirements"
            },
            "Ridge Regression": {
                "emoji": "📏", "description": "Linear regression with L2 regularization to prevent overfitting.",
                "strengths": ["Handles multicollinearity", "Prevents overfitting"], "best_for": "Linear problems with many features"
            },
            "Lasso Regression": {
                "emoji": "🎯", "description": "Linear regression with L1 regularization for feature selection.",
                "strengths": ["Feature selection", "Prevents overfitting"], "best_for": "Sparse feature selection"
            },
            "Neural Network": {
                "emoji": "🧠", "description": "Multi-layer perceptron for regression tasks.",
                "strengths": ["Learns complex patterns", "Flexible"], "best_for": "Complex patterns, large datasets"
            }
        }


def create_model(model_name, problem_type, hyperparams):
    """
    Create and configure a model instance.
    
    Args:
        model_name (str): Name of the model
        problem_type (str): Either "Classification" or "Regression"
        hyperparams (dict): Hyperparameters for the model
        
    Returns:
        sklearn model: Configured model instance
    """
    params = hyperparams.get(model_name, {})
    
    # Classification Models
    if problem_type == "Classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                solver=params.get('solver', 'lbfgs'),
                random_state=42
            )
        elif model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42
            )
        elif model_name == "Support Vector Machine":
            return SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                gamma=params.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                criterion=params.get('criterion', 'gini'),
                random_state=42
            )
        elif model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier(
                n_neighbors=params.get('n_neighbors', 5),
                weights=params.get('weights', 'uniform'),
                metric=params.get('metric', 'euclidean')
            )
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                random_state=42
            )
        elif model_name == "AdaBoost":
            return AdaBoostClassifier(
                n_estimators=params.get('n_estimators', 50),
                learning_rate=params.get('learning_rate', 1.0),
                algorithm=params.get('algorithm', 'SAMME'),
                random_state=42
            )
        elif model_name == "Naive Bayes":
            return GaussianNB(
                var_smoothing=params.get('var_smoothing', 1e-9)
            )
        elif model_name == "Neural Network":
            return MLPClassifier(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                learning_rate_init=params.get('learning_rate_init', 0.001),
                max_iter=params.get('max_iter', 200),
                activation=params.get('activation', 'relu'),
                random_state=42
            )
        elif model_name == "Ridge Classifier":
            return RidgeClassifier(
                alpha=params.get('alpha', 1.0),
                random_state=42
            )
            
    # Regression Models
    else:  # Regression
        if model_name == "Linear Regression":
            return LinearRegression(
                fit_intercept=params.get('fit_intercept', True)
            )
        elif model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42
            )
        elif model_name == "Support Vector Regression":
            return SVR(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                epsilon=params.get('epsilon', 0.1)
            )
        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42
            )
        elif model_name == "K-Nearest Neighbors":
            return KNeighborsRegressor(
                n_neighbors=params.get('n_neighbors', 5),
                weights=params.get('weights', 'uniform'),
                metric=params.get('metric', 'euclidean')
            )
        elif model_name == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                random_state=42
            )
        elif model_name == "Ridge Regression":
            return Ridge(
                alpha=params.get('alpha', 1.0),
                random_state=42
            )
        elif model_name == "Lasso Regression":
            return Lasso(
                alpha=params.get('alpha', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=42
            )
        elif model_name == "Neural Network":
            return MLPRegressor(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                learning_rate_init=params.get('learning_rate_init', 0.001),
                max_iter=params.get('max_iter', 200),
                activation=params.get('activation', 'relu'),
                random_state=42
            )
    
    raise ValueError(f"Unknown model: {model_name} for {problem_type}")


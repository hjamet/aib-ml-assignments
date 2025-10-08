import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="ML Demo: Interactive Machine Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def load_titanic_data():
    """Load and prepare the Titanic dataset"""
    # Try to load from seaborn first
    df = sns.load_dataset('titanic')
    print(df.head())
    return df

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Machine Learning Adventure: Interactive Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎓 Welcome to Your ML Journey!</h3>
        <p>This interactive demo will teach you Machine Learning concepts without any coding! 
        Simply use the controls in the sidebar to explore, learn, and experiment.</p>
        <p><strong>What you'll learn:</strong> Data exploration, preprocessing, model training, and evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_titanic_data()
    
    # Sidebar for user controls
    st.sidebar.markdown("# 🎛️ Control Panel")
    st.sidebar.markdown("---")
    
    # Problem type selection
    problem_type = st.sidebar.radio(
        "🎯 Choose Problem Type:",
        ["Classification", "Regression"],
        help="Classification predicts categories (survived/not survived). Regression predicts continuous values (age, fare)."
    )
    
    # Dataset overview
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        st.metric("Survival Rate", f"{df['survived'].mean():.1%}")
    with col3:
        st.metric("Features", df.shape[1])
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Show sample data
    if st.checkbox("📋 Show Sample Data", value=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Section 1: Data Exploration
    st.markdown('<h2 class="section-header">🔍 Interactive Data Exploration</h2>', unsafe_allow_html=True)
    
    exploration_feature = st.sidebar.selectbox(
        "🔍 Explore Feature:",
        ["sex", "pclass", "age", "fare", "embarked"],
        format_func=lambda x: {
            "sex": "👥 Gender", 
            "pclass": "🎫 Passenger Class",
            "age": "👶 Age",
            "fare": "💰 Fare",
            "embarked": "🚢 Port of Embarkation"
        }[x]
    )
    
    # Create exploration plots
    col1, col2 = st.columns(2)
    
    with col1:
        if exploration_feature in ['age', 'fare']:
            fig = px.histogram(df, x=exploration_feature, title=f'Distribution of {exploration_feature.title()}',
                             color_discrete_sequence=['skyblue'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(df, x=exploration_feature, title=f'Distribution of {exploration_feature.title()}',
                             color_discrete_sequence=['lightgreen'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        survival_by_feature = df.groupby(exploration_feature)['survived'].mean().reset_index()
        fig = px.bar(survival_by_feature, x=exploration_feature, y='survived',
                    title=f'Survival Rate by {exploration_feature.title()}',
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
    
    st.info(f"💡 **Insight:** {insights[exploration_feature]}")
    
    # Section 2: Data Preprocessing
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🔧 Preprocessing Options")
    
    # Preprocessing controls
    missing_age_option = st.sidebar.selectbox(
        "Handle Missing Ages:",
        ["Fill with median", "Fill with mean", "Drop rows"]
    )
    
    normalize_features = st.sidebar.checkbox("📏 Normalize Features", value=True)
    
    selected_features = st.sidebar.multiselect(
        "📊 Select Features:",
        ["Age", "Sex", "Passenger Class", "Fare", "Siblings/Spouses", "Parents/Children", "Port of Embarkation"],
        default=["Age", "Sex", "Passenger Class", "Fare"]
    )
    
    if selected_features:
        # Preprocess data
        processed_df = df.copy()
        
        # Handle missing ages
        if missing_age_option == "Fill with median":
            processed_df['age'].fillna(processed_df['age'].median(), inplace=True)
        elif missing_age_option == "Fill with mean":
            processed_df['age'].fillna(processed_df['age'].mean(), inplace=True)
        
        # Encode categorical variables
        processed_df['sex_encoded'] = (processed_df['sex'] == 'male').astype(int)
        processed_df['embarked_encoded'] = LabelEncoder().fit_transform(processed_df['embarked'].fillna('S'))
        
        # Feature mapping
        feature_mapping = {
            'Age': 'age',
            'Sex': 'sex_encoded',
            'Passenger Class': 'pclass',
            'Fare': 'fare',
            'Siblings/Spouses': 'sibsp',
            'Parents/Children': 'parch',
            'Port of Embarkation': 'embarked_encoded'
        }
        
        feature_names = [feature_mapping[f] for f in selected_features]
        X = processed_df[feature_names].copy()
        
        # Drop rows with missing values
        before_count = len(X)
        X = X.dropna()
        y = processed_df.loc[X.index, 'survived']
        after_count = len(X)
        
        # Normalize if requested
        if normalize_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Show preprocessing results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Samples", before_count)
        with col2:
            st.metric("Final Samples", after_count)
        with col3:
            st.metric("Features Used", len(selected_features))
        
        if before_count != after_count:
            st.warning(f"⚠️ Removed {before_count - after_count} rows with missing values")
        
        st.success(f"✅ Data preprocessing completed! Using {len(selected_features)} features: {', '.join(selected_features)}")
        
        # Section 3: Model Training
        st.markdown('<h2 class="section-header">🤖 Model Training & Comparison</h2>', unsafe_allow_html=True)

        
        st.sidebar.markdown("### 🤖 Model Selection")
        
        # Model selection based on problem type - only show appropriate models
        if problem_type == "Classification":
            available_models = {
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
            default_models = ["Random Forest", "Logistic Regression", "Support Vector Machine"]
        else:  # Regression
            available_models = {
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
            default_models = ["Linear Regression", "Random Forest", "Ridge Regression"]
        
        selected_models = st.sidebar.multiselect(
            "Choose Models to Compare:",
            list(available_models.keys()),
            default=default_models,
            help="Select multiple models to compare their performance"
        )
        
        # Show model descriptions
        if selected_models:
            st.sidebar.markdown("### 📚 Selected Models")
            for model_name in selected_models[:3]:  # Show max 3 descriptions
                st.sidebar.markdown(f"**{available_models[model_name]}**")
        
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20, 5)
        
        # Dynamic hyperparameter controls based on selected models
        st.sidebar.markdown("### ⚙️ Hyperparameter Tuning")
        st.sidebar.info("💡 Hyperparameters control how models learn. Experiment with different values!")
        hyperparams = {}
        
        for model_name in selected_models:
            # Only show hyperparameters for models available in current problem type
            if model_name not in available_models:
                continue
                
            st.sidebar.markdown(f"**{model_name}**")
            
            if model_name == "Logistic Regression":
                hyperparams[model_name] = {
                    'C': st.sidebar.slider(f"LR - Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"lr_c", 
                                         help="Higher C = less regularization (more complex model)"),
                    'max_iter': st.sidebar.slider(f"LR - Max Iterations", 100, 2000, 1000, 100, key=f"lr_iter",
                                                help="Maximum number of training iterations"),
                    'solver': st.sidebar.selectbox(f"LR - Solver", ['liblinear', 'lbfgs', 'saga'], index=1, key=f"lr_solver",
                                                  help="Algorithm to use for optimization")
                }
                
            elif model_name == "Random Forest":
                hyperparams[model_name] = {
                    'n_estimators': st.sidebar.slider(f"RF - Number of Trees", 10, 500, 100, 10, key=f"rf_trees",
                                                     help="More trees = better performance but slower training"),
                    'max_depth': st.sidebar.slider(f"RF - Max Depth", 3, 20, 10, 1, key=f"rf_depth",
                                                   help="Maximum depth of trees. Deeper = more complex"),
                    'min_samples_split': st.sidebar.slider(f"RF - Min Samples Split", 2, 20, 2, 1, key=f"rf_split",
                                                          help="Minimum samples required to split a node"),
                    'min_samples_leaf': st.sidebar.slider(f"RF - Min Samples Leaf", 1, 10, 1, 1, key=f"rf_leaf",
                                                         help="Minimum samples required at each leaf node")
                }
                
            elif model_name == "Support Vector Machine":
                hyperparams[model_name] = {
                    'C': st.sidebar.slider(f"SVM - Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"svm_c"),
                    'kernel': st.sidebar.selectbox(f"SVM - Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], key=f"svm_kernel"),
                    'gamma': st.sidebar.selectbox(f"SVM - Gamma", ['scale', 'auto'], key=f"svm_gamma")
                }
                
            elif model_name == "Decision Tree":
                hyperparams[model_name] = {
                    'max_depth': st.sidebar.slider(f"DT - Max Depth", 3, 20, 10, 1, key=f"dt_depth"),
                    'min_samples_split': st.sidebar.slider(f"DT - Min Samples Split", 2, 20, 2, 1, key=f"dt_split"),
                    'min_samples_leaf': st.sidebar.slider(f"DT - Min Samples Leaf", 1, 10, 1, 1, key=f"dt_leaf"),
                    'criterion': st.sidebar.selectbox(f"DT - Criterion", ['gini', 'entropy'], key=f"dt_criterion")
                }
                
            elif model_name == "K-Nearest Neighbors":
                hyperparams[model_name] = {
                    'n_neighbors': st.sidebar.slider(f"KNN - Number of Neighbors", 1, 20, 5, 1, key=f"knn_n"),
                    'weights': st.sidebar.selectbox(f"KNN - Weights", ['uniform', 'distance'], key=f"knn_weights"),
                    'metric': st.sidebar.selectbox(f"KNN - Distance Metric", ['euclidean', 'manhattan', 'minkowski'], key=f"knn_metric")
                }
                
            elif model_name == "Gradient Boosting":
                hyperparams[model_name] = {
                    'n_estimators': st.sidebar.slider(f"GB - Number of Estimators", 50, 300, 100, 10, key=f"gb_trees"),
                    'learning_rate': st.sidebar.slider(f"GB - Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"gb_lr"),
                    'max_depth': st.sidebar.slider(f"GB - Max Depth", 3, 10, 6, 1, key=f"gb_depth")
                }
                
            elif model_name == "AdaBoost":
                hyperparams[model_name] = {
                    'n_estimators': st.sidebar.slider(f"Ada - Number of Estimators", 10, 200, 50, 10, key=f"ada_trees"),
                    'learning_rate': st.sidebar.slider(f"Ada - Learning Rate", 0.1, 2.0, 1.0, 0.1, key=f"ada_lr"),
                    'algorithm': st.sidebar.selectbox(f"Ada - Algorithm", ['SAMME', 'SAMME.R'], key=f"ada_algo")
                }
                
            elif model_name == "Naive Bayes":
                hyperparams[model_name] = {
                    'var_smoothing': st.sidebar.slider(f"NB - Smoothing", 1e-10, 1e-5, 1e-9, 1e-10, key=f"nb_smooth", format="%.2e")
                }
                
            elif model_name == "Neural Network":
                layer_sizes = st.sidebar.multiselect(f"NN - Hidden Layer Sizes", [50, 100, 200, 300], default=[100], key=f"nn_layers")
                hyperparams[model_name] = {
                    'hidden_layer_sizes': tuple(layer_sizes) if layer_sizes else (100,),
                    'learning_rate_init': st.sidebar.slider(f"NN - Learning Rate", 0.001, 0.1, 0.001, 0.001, key=f"nn_lr"),
                    'max_iter': st.sidebar.slider(f"NN - Max Iterations", 100, 1000, 200, 50, key=f"nn_iter"),
                    'activation': st.sidebar.selectbox(f"NN - Activation", ['relu', 'tanh', 'logistic'], key=f"nn_activation")
                }
                
            elif model_name == "Ridge Classifier":
                hyperparams[model_name] = {
                    'alpha': st.sidebar.slider(f"Ridge - Alpha (Regularization)", 0.1, 10.0, 1.0, 0.1, key=f"ridge_alpha"),
                    'solver': st.sidebar.selectbox(f"Ridge - Solver", ['auto', 'svd', 'cholesky', 'lsqr'], key=f"ridge_solver")
                }
                
            # Regression Models
            elif model_name == "Linear Regression":
                hyperparams[model_name] = {
                    'fit_intercept': st.sidebar.checkbox(f"LinReg - Fit Intercept", value=True, key=f"linreg_intercept"),
                    'normalize': st.sidebar.checkbox(f"LinReg - Normalize", value=False, key=f"linreg_normalize")
                }
                
            elif model_name == "Support Vector Regression":
                hyperparams[model_name] = {
                    'C': st.sidebar.slider(f"SVR - Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"svr_c"),
                    'kernel': st.sidebar.selectbox(f"SVR - Kernel", ['rbf', 'linear', 'poly'], key=f"svr_kernel"),
                    'epsilon': st.sidebar.slider(f"SVR - Epsilon", 0.01, 1.0, 0.1, 0.01, key=f"svr_epsilon")
                }
                
            elif model_name == "Ridge Regression":
                hyperparams[model_name] = {
                    'alpha': st.sidebar.slider(f"Ridge - Alpha", 0.1, 10.0, 1.0, 0.1, key=f"ridge_reg_alpha"),
                    'solver': st.sidebar.selectbox(f"Ridge - Solver", ['auto', 'svd', 'cholesky', 'lsqr'], key=f"ridge_reg_solver")
                }
                
            elif model_name == "Lasso Regression":
                hyperparams[model_name] = {
                    'alpha': st.sidebar.slider(f"Lasso - Alpha", 0.01, 2.0, 1.0, 0.01, key=f"lasso_alpha"),
                    'max_iter': st.sidebar.slider(f"Lasso - Max Iterations", 100, 2000, 1000, 100, key=f"lasso_iter")
                }
        
        # Store target information for later use (already defined above)
        
        if selected_models:
            # Show detailed model information
            with st.expander("📚 Learn About Your Selected Models", expanded=False):
                # Define model descriptions based on problem type
                if problem_type == "Classification":
                    model_descriptions = {
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
                    model_descriptions = {
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
                            "emoji": "�", "description": "Predicts values based on average of k nearest neighbors.",
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
                
                cols = st.columns(min(len(selected_models), 3))
                for i, model_name in enumerate(selected_models):
                    with cols[i % 3]:
                        info = model_descriptions[model_name]
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{info['emoji']} {model_name}</h4>
                            <p><strong>Description:</strong> {info['description']}</p>
                            <p><strong>Best for:</strong> {info['best_for']}</p>
                            <p><strong>Key strengths:</strong> {', '.join(info['strengths'][:2])}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Train models
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y
            )
            
            models = {}
            results = []
            
            for model_name in selected_models:
                params = hyperparams.get(model_name, {})
                
                # Classification Models
                if problem_type == "Classification":
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(
                            C=params.get('C', 1.0),
                            max_iter=params.get('max_iter', 1000),
                            solver=params.get('solver', 'lbfgs'),
                            random_state=42
                        )
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=params.get('n_estimators', 100),
                            max_depth=params.get('max_depth', 10),
                            min_samples_split=params.get('min_samples_split', 2),
                            min_samples_leaf=params.get('min_samples_leaf', 1),
                            random_state=42
                        )
                    elif model_name == "Support Vector Machine":
                        model = SVC(
                            C=params.get('C', 1.0),
                            kernel=params.get('kernel', 'rbf'),
                            gamma=params.get('gamma', 'scale'),
                            probability=True,
                            random_state=42
                        )
                    elif model_name == "Decision Tree":
                        model = DecisionTreeClassifier(
                            max_depth=params.get('max_depth', 10),
                            min_samples_split=params.get('min_samples_split', 2),
                            min_samples_leaf=params.get('min_samples_leaf', 1),
                            criterion=params.get('criterion', 'gini'),
                            random_state=42
                        )
                    elif model_name == "K-Nearest Neighbors":
                        model = KNeighborsClassifier(
                            n_neighbors=params.get('n_neighbors', 5),
                            weights=params.get('weights', 'uniform'),
                            metric=params.get('metric', 'euclidean')
                        )
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingClassifier(
                            n_estimators=params.get('n_estimators', 100),
                            learning_rate=params.get('learning_rate', 0.1),
                            max_depth=params.get('max_depth', 6),
                            random_state=42
                        )
                    elif model_name == "AdaBoost":
                        model = AdaBoostClassifier(
                            n_estimators=params.get('n_estimators', 50),
                            learning_rate=params.get('learning_rate', 1.0),
                            algorithm=params.get('algorithm', 'SAMME'),
                            random_state=42
                        )
                    elif model_name == "Naive Bayes":
                        model = GaussianNB(
                            var_smoothing=params.get('var_smoothing', 1e-9)
                        )
                    elif model_name == "Neural Network":
                        model = MLPClassifier(
                            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                            learning_rate_init=params.get('learning_rate_init', 0.001),
                            max_iter=params.get('max_iter', 200),
                            activation=params.get('activation', 'relu'),
                            random_state=42
                        )
                    elif model_name == "Ridge Classifier":
                        model = RidgeClassifier(
                            alpha=params.get('alpha', 1.0),
                            solver=params.get('solver', 'auto'),
                            random_state=42
                        )
                        
                # Regression Models
                else:  # Regression
                    if model_name == "Linear Regression":
                        model = LinearRegression(
                            fit_intercept=params.get('fit_intercept', True)
                        )
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(
                            n_estimators=params.get('n_estimators', 100),
                            max_depth=params.get('max_depth', 10),
                            min_samples_split=params.get('min_samples_split', 2),
                            min_samples_leaf=params.get('min_samples_leaf', 1),
                            random_state=42
                        )
                    elif model_name == "Support Vector Regression":
                        model = SVR(
                            C=params.get('C', 1.0),
                            kernel=params.get('kernel', 'rbf'),
                            epsilon=params.get('epsilon', 0.1)
                        )
                    elif model_name == "Decision Tree":
                        model = DecisionTreeRegressor(
                            max_depth=params.get('max_depth', 10),
                            min_samples_split=params.get('min_samples_split', 2),
                            min_samples_leaf=params.get('min_samples_leaf', 1),
                            random_state=42
                        )
                    elif model_name == "K-Nearest Neighbors":
                        model = KNeighborsRegressor(
                            n_neighbors=params.get('n_neighbors', 5),
                            weights=params.get('weights', 'uniform'),
                            metric=params.get('metric', 'euclidean')
                        )
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingRegressor(
                            n_estimators=params.get('n_estimators', 100),
                            learning_rate=params.get('learning_rate', 0.1),
                            max_depth=params.get('max_depth', 6),
                            random_state=42
                        )
                    elif model_name == "Ridge Regression":
                        model = Ridge(
                            alpha=params.get('alpha', 1.0),
                            solver=params.get('solver', 'auto'),
                            random_state=42
                        )
                    elif model_name == "Lasso Regression":
                        model = Lasso(
                            alpha=params.get('alpha', 1.0),
                            max_iter=params.get('max_iter', 1000),
                            random_state=42
                        )
                    elif model_name == "Neural Network":
                        model = MLPRegressor(
                            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                            learning_rate_init=params.get('learning_rate_init', 0.001),
                            max_iter=params.get('max_iter', 200),
                            activation=params.get('activation', 'relu'),
                            random_state=42
                        )
                
                # Train model
                model.fit(X_train, y_train)
                models[model_name] = model
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics based on problem type
                if problem_type == "Classification":
                    train_score = accuracy_score(y_train, train_pred)
                    test_score = accuracy_score(y_test, test_pred)
                    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
                    
                    results.append({
                        'Model': model_name,
                        'Training Score': train_score,
                        'Test Score': test_score,
                        'Precision': precision,
                        'Recall': recall,
                        'Overfitting': train_score - test_score,
                        'Metric': 'Accuracy'
                    })
                else:  # Regression
                    train_score = r2_score(y_train, train_pred)
                    test_score = r2_score(y_test, test_pred)
                    mse = mean_squared_error(y_test, test_pred)
                    mae = mean_absolute_error(y_test, test_pred)
                    rmse = np.sqrt(mse)
                    
                    results.append({
                        'Model': model_name,
                        'Training Score': train_score,
                        'Test Score': test_score,
                        'MSE': mse,
                        'MAE': mae,
                        'RMSE': rmse,
                        'Overfitting': train_score - test_score,
                        'Metric': 'R² Score'
                    })
            
            # Display results
            results_df = pd.DataFrame(results)
            
            st.subheader("📊 Model Performance Comparison")
            
            # Metrics display
            for i, result in enumerate(results):
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
            
            # Comparison chart
            metric_name = results[0]['Metric']
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f'{metric_name} Comparison', 'Overfitting Analysis'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Score comparison
            fig.add_trace(
                go.Bar(name=f'Training {metric_name}', x=results_df['Model'], y=results_df['Training Score'],
                      marker_color='lightblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name=f'Test {metric_name}', x=results_df['Model'], y=results_df['Test Score'],
                      marker_color='lightcoral'),
                row=1, col=1
            )
            
            # Overfitting analysis
            colors = ['red' if x > 0.05 else 'orange' if x > 0.02 else 'green' for x in results_df['Overfitting']]
            fig.add_trace(
                go.Bar(name='Overfitting Score', x=results_df['Model'], y=results_df['Overfitting'],
                      marker_color=colors, showlegend=False),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model selection for detailed analysis
            best_model_name = results_df.loc[results_df['Test Score'].idxmax(), 'Model']
            best_model = models[best_model_name]
            
            # 2D Feature Visualization
            if len(feature_names) >= 2:
                st.subheader("📊 2D Feature Visualization")
                
                # Feature selection for 2D plot
                col1, col2 = st.columns(2)
                with col1:
                    feature_x = st.selectbox("Choose X-axis feature:", feature_names, key="2d_x")
                with col2:
                    feature_y = st.selectbox("Choose Y-axis feature:", 
                                           [f for f in feature_names if f != feature_x], key="2d_y")
                
                if len(feature_names) == 2:
                    st.info("� **Perfect!** With exactly 2 features, you can see how the classifier creates decision boundaries!")
                    
                    # Create decision boundary plot for classification
                    if problem_type == "Classification":
                        fig = create_decision_boundary_plot(X_train, y_train, best_model, feature_x, feature_y, best_model_name)
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("💡 **Decision Boundary**: The colored regions show where the model predicts each class. Points show actual data.")
                    else:
                        # For regression, show prediction surface
                        fig = create_regression_surface_plot(X_train, y_train, best_model, feature_x, feature_y, best_model_name, "Target")
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("💡 **Prediction Surface**: The colored surface shows the model's predictions across the feature space.")
                else:
                    # Show 2D scatter plot with predictions
                    fig = create_2d_scatter_plot(X_test, y_test, test_pred, feature_x, feature_y, problem_type, "Target")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if problem_type == "Classification":
                        st.info("💡 **2D Feature Plot**: Each point is a passenger. Colors show actual vs predicted classes. Look for patterns!")
                    else:
                        st.info("💡 **2D Feature Plot**: Each point is a passenger. Color intensity shows the target value. Size shows prediction accuracy.")
            
            st.subheader(f"�🏆 Best Model: {best_model_name}")
            
            # Classification specific visualizations
            if problem_type == "Classification":
                # Confusion Matrix
                y_pred_best = best_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred_best)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.imshow(cm, 
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Did not survive', 'Survived'],
                                   y=['Did not survive', 'Survived'],
                                   color_continuous_scale='Blues',
                                   title="Confusion Matrix")
                    fig.update_traces(text=cm, texttemplate="%{text}")
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Regression specific visualizations
                y_pred_best = best_model.predict(X_test)
                st.subheader("📈 Regression Analysis")
                
                # Actual vs Predicted plot
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(x=y_test, y=y_pred_best, 
                                   title="Actual vs Predicted Values",
                                   labels={'x': 'Actual Values', 'y': 'Predicted Values'})
                    # Add perfect prediction line
                    min_val, max_val = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
                    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                line=dict(color="red", width=2, dash="dash"))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Residuals plot
                    residuals = y_test - y_pred_best
                    fig = px.scatter(x=y_pred_best, y=residuals,
                                   title="Residuals Plot",
                                   labels={'x': 'Predicted Values', 'y': 'Residuals'})
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Regression metrics explanation
                mse = mean_squared_error(y_test, y_pred_best)
                mae = mean_absolute_error(y_test, y_pred_best)
                r2 = r2_score(y_test, y_pred_best)
                
                st.markdown(f"""
                **📊 Regression Metrics Explained:**
                - **R² Score: {r2:.3f}** - Proportion of variance explained (1.0 = perfect fit)
                - **RMSE: {np.sqrt(mse):.2f}** - Average prediction error in original units
                - **MAE: {mae:.2f}** - Mean absolute error (robust to outliers)
                """)
            
            # Continue with feature importance section (works for both classification and regression)
            if problem_type == "Classification":
                # Confusion matrix explanation  
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
            
            # Feature Importance (for tree-based models)
            if hasattr(best_model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                           title=f'Which Features Matter Most? ({best_model_name})',
                           color_discrete_sequence=['lightgreen'])
                st.plotly_chart(fig, width='stretch')
                
                st.info(f"💡 **Feature Importance** shows which passenger characteristics the {best_model_name} considers most important for predicting survival.")
            
            elif hasattr(best_model, 'coef_'):
                st.subheader("🎯 Feature Coefficients")
                
                # For linear models, show coefficients
                feature_coef = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': best_model.coef_[0] if len(best_model.coef_.shape) > 1 else best_model.coef_
                }).sort_values('coefficient', ascending=True)
                
                fig = px.bar(feature_coef, x='coefficient', y='feature', orientation='h',
                           title=f'Feature Coefficients ({best_model_name})',
                           color='coefficient', color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, width='stretch')
                
                st.info(f"💡 **Feature Coefficients** show how much each feature influences the {best_model_name}'s predictions. Positive values increase survival probability, negative values decrease it.")
            
            # Interactive Prediction
            st.markdown('<h2 class="section-header">🔮 Make Your Own Predictions</h2>', unsafe_allow_html=True)
            
            st.markdown("Try different passenger profiles and see what the AI predicts!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Passenger Profile")
                
                # Create input widgets based on selected features
                prediction_inputs = {}
                
                if 'Age' in selected_features:
                    prediction_inputs['age'] = st.slider("Age", 0, 80, 30)
                if 'Sex' in selected_features:
                    sex_input = st.selectbox("Gender", ["Female", "Male"])
                    prediction_inputs['sex_encoded'] = 1 if sex_input == "Male" else 0
                if 'Passenger Class' in selected_features:
                    prediction_inputs['pclass'] = st.selectbox("Passenger Class", [1, 2, 3], index=1)
                if 'Fare' in selected_features:
                    prediction_inputs['fare'] = st.slider("Fare ($)", 0, 500, 50)
                if 'Siblings/Spouses' in selected_features:
                    prediction_inputs['sibsp'] = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
                if 'Parents/Children' in selected_features:
                    prediction_inputs['parch'] = st.slider("Parents/Children Aboard", 0, 6, 0)
                if 'Port of Embarkation' in selected_features:
                    port_input = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
                    port_mapping = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
                    prediction_inputs['embarked_encoded'] = port_mapping[port_input]
            
            with col2:
                st.subheader("🎯 Prediction Result")
                
                predict_button_text = "🔮 Predict Survival" if problem_type == "Classification" else "🔮 Predict Value"
                if st.button(predict_button_text, type="primary"):
                    # Prepare input for prediction
                    input_data = []
                    for feature_name in feature_names:
                        if feature_name in prediction_inputs:
                            input_data.append(prediction_inputs[feature_name])
                        else:
                            # Use median values for missing features
                            input_data.append(X[feature_name].median())
                    
                    input_array = np.array(input_data).reshape(1, -1)
                    
                    # Normalize if needed
                    if normalize_features:
                        input_array = scaler.transform(input_array)
                    
                    # Make prediction
                    prediction = best_model.predict(input_array)[0]
                    
                    if problem_type == "Classification":
                        probability = best_model.predict_proba(input_array)[0]
                        
                        if prediction == 1:
                            st.success(f"🎉 **SURVIVAL PREDICTED!**")
                            st.success(f"Confidence: {probability[1]:.1%}")
                        else:
                            st.error(f"💔 **Did not survive**")
                            st.error(f"Confidence: {probability[0]:.1%}")
                        
                        # Show probability breakdown
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
    
    # Learning Summary
    st.markdown('<h2 class="section-header">🎓 What You\'ve Learned</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <h3>🌟 Congratulations! You've mastered key ML concepts:</h3>
        <ul>
            <li><strong>Data Exploration:</strong> Understanding patterns in real-world data</li>
            <li><strong>Data Preprocessing:</strong> Preparing data for machine learning algorithms</li>
            <li><strong>Model Selection:</strong> Choosing the right algorithm for your problem</li>
            <li><strong>Model Evaluation:</strong> Understanding accuracy, precision, recall, and confusion matrices</li>
            <li><strong>Overfitting:</strong> Balancing model complexity for real-world performance</li>
            <li><strong>Feature Importance:</strong> Identifying which factors matter most</li>
        </ul>
        
        🚀 Real-world applications include:
        Medical diagnosis, fraud detection, recommendation systems, autonomous vehicles, 
        weather prediction, and much more!
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
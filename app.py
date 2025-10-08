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
    
    # Sidebar navigation
    st.sidebar.markdown("# 🧭 Navigation")
    page = st.sidebar.radio(
        "Choisir une page:",
        ["📊 Preprocessing & Exploration", "📈 Régression", "🎯 Classification"],
        key="navigation_page"
    )
    
    # =============================================
    # PAGE 1: PREPROCESSING & EXPLORATION
    # =============================================
    if page == "📊 Preprocessing & Exploration":
        st.markdown('<h2 class="section-header">🔍 Interactive Data Exploration</h2>', unsafe_allow_html=True)
        
        exploration_feature = st.selectbox(
            "🔍 Explore Feature:",
            ["sex", "pclass", "age", "fare", "embarked"],
            format_func=lambda x: {
                "sex": "👥 Gender", 
                "pclass": "🎫 Passenger Class",
                "age": "👶 Age",
                "fare": "💰 Fare",
                "embarked": "🚢 Port of Embarkation"
            }[x],
            key="exploration_feature"
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
        
        # Preprocessing controls
        st.selectbox(
            "Handle Missing Ages:",
            ["Fill with median", "Fill with mean", "Drop rows"],
            key="missing_age_option"
        )
        
        st.checkbox("📏 Normalize Features", value=True, key="normalize_features")
        
        st.multiselect(
            "📊 Select Features:",
            ["Age", "Sex", "Passenger Class", "Fare", "Siblings/Spouses", "Parents/Children", "Port of Embarkation"],
            default=["Age", "Sex", "Passenger Class", "Fare"],
            key="selected_features"
        )
        
        st.slider("Test Set Size (%)", 10, 40, 20, 5, key="test_size")
        
        if st.session_state.selected_features:
            # Get values from session_state (set by widgets)
            selected_features = st.session_state.selected_features
            normalize_features = st.session_state.normalize_features
            missing_age_option = st.session_state.missing_age_option
            
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
            scaler = None
            if normalize_features:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Store processed data in session state
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.feature_names = feature_names
            st.session_state.scaler = scaler
            
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
    
    # =============================================
    # PAGE 2: REGRESSION
    # =============================================
    elif page == "📈 Régression":
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
        selected_features = st.session_state.get('selected_features', [])
        
        st.markdown('<h2 class="section-header">🤖 Regression Models Training & Comparison</h2>', unsafe_allow_html=True)
        
        # Model selection in page
        st.markdown("### 📋 Model Selection")
        
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
        
        selected_models = st.multiselect(
            "Choose Regression Models:",
            list(available_models.keys()),
            default=["Linear Regression", "Random Forest", "Ridge Regression"],
            help="Select multiple models to compare their performance",
            key="regression_models"
        )
        
        # Get test size from session state
        test_size = st.session_state.get("test_size", 20)
        
        # Hyperparameters in page
        st.markdown("### ⚙️ Hyperparameter Tuning")
        st.info("💡 Hyperparameters control how models learn. Experiment with different values!")
        hyperparams = {}
        
        for model_name in selected_models:
            with st.expander(f"🔧 {model_name} - Hyperparameters"):
                if model_name == "Linear Regression":
                    hyperparams[model_name] = {
                        'fit_intercept': st.checkbox(f"Fit Intercept", value=True, key=f"reg_linreg_intercept"),
                    }
                    
                elif model_name == "Random Forest":
                    hyperparams[model_name] = {
                        'n_estimators': st.slider(f"Number of Trees", 10, 500, 100, 10, key=f"reg_rf_trees"),
                        'max_depth': st.slider(f"Max Depth", 3, 20, 10, 1, key=f"reg_rf_depth"),
                        'min_samples_split': st.slider(f"Min Samples Split", 2, 20, 2, 1, key=f"reg_rf_split"),
                        'min_samples_leaf': st.slider(f"Min Samples Leaf", 1, 10, 1, 1, key=f"reg_rf_leaf")
                    }
                    
                elif model_name == "Support Vector Regression":
                    hyperparams[model_name] = {
                        'C': st.slider(f"Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"reg_svr_c"),
                        'kernel': st.selectbox(f"Kernel", ['rbf', 'linear', 'poly'], key=f"reg_svr_kernel"),
                        'epsilon': st.slider(f"Epsilon", 0.01, 1.0, 0.1, 0.01, key=f"reg_svr_epsilon")
                    }
                    
                elif model_name == "Decision Tree":
                    hyperparams[model_name] = {
                        'max_depth': st.slider(f"Max Depth", 3, 20, 10, 1, key=f"reg_dt_depth"),
                        'min_samples_split': st.slider(f"Min Samples Split", 2, 20, 2, 1, key=f"reg_dt_split"),
                        'min_samples_leaf': st.slider(f"Min Samples Leaf", 1, 10, 1, 1, key=f"reg_dt_leaf")
                    }
                    
                elif model_name == "K-Nearest Neighbors":
                    hyperparams[model_name] = {
                        'n_neighbors': st.slider(f"Number of Neighbors", 1, 20, 5, 1, key=f"reg_knn_n"),
                        'weights': st.selectbox(f"Weights", ['uniform', 'distance'], key=f"reg_knn_weights"),
                        'metric': st.selectbox(f"Distance Metric", ['euclidean', 'manhattan', 'minkowski'], key=f"reg_knn_metric")
                    }
                    
                elif model_name == "Gradient Boosting":
                    hyperparams[model_name] = {
                        'n_estimators': st.slider(f"Number of Estimators", 50, 300, 100, 10, key=f"reg_gb_trees"),
                        'learning_rate': st.slider(f"Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"reg_gb_lr"),
                        'max_depth': st.slider(f"Max Depth", 3, 10, 6, 1, key=f"reg_gb_depth")
                    }
                    
                elif model_name == "Ridge Regression":
                    hyperparams[model_name] = {
                        'alpha': st.slider(f"Alpha", 0.1, 10.0, 1.0, 0.1, key=f"reg_ridge_alpha"),
                    }
                    
                elif model_name == "Lasso Regression":
                    hyperparams[model_name] = {
                        'alpha': st.slider(f"Alpha", 0.01, 2.0, 1.0, 0.01, key=f"reg_lasso_alpha"),
                        'max_iter': st.slider(f"Max Iterations", 100, 2000, 1000, 100, key=f"reg_lasso_iter")
                    }
                    
                elif model_name == "Neural Network":
                    layer_sizes = st.multiselect(f"Hidden Layer Sizes", [50, 100, 200, 300], default=[100], key=f"reg_nn_layers")
                    hyperparams[model_name] = {
                        'hidden_layer_sizes': tuple(layer_sizes) if layer_sizes else (100,),
                        'learning_rate_init': st.slider(f"Learning Rate", 0.001, 0.1, 0.001, 0.001, key=f"reg_nn_lr"),
                        'max_iter': st.slider(f"Max Iterations", 100, 1000, 200, 50, key=f"reg_nn_iter"),
                        'activation': st.selectbox(f"Activation", ['relu', 'tanh', 'logistic'], key=f"reg_nn_activation")
                    }
        
        if selected_models:
            # Train models
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
            )
            
            models = {}
            results = []
            
            for model_name in selected_models:
                params = hyperparams.get(model_name, {})
                
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
                
                # Calculate metrics
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
            for result in results:
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
            colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in results_df['Overfitting']]
            fig.add_trace(
                go.Bar(name='Overfitting Score', x=results_df['Model'], y=results_df['Overfitting'],
                      marker_color=colors, showlegend=False),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model
            best_model_name = results_df.loc[results_df['Test Score'].idxmax(), 'Model']
            best_model = models[best_model_name]
            y_pred_best = best_model.predict(X_test)
            
            st.subheader(f"🏆 Best Model: {best_model_name}")
            
            # Regression visualizations
            st.markdown("### 📈 Regression Analysis")
            
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
            
            # Metrics explanation
            mse = mean_squared_error(y_test, y_pred_best)
            mae = mean_absolute_error(y_test, y_pred_best)
            r2 = r2_score(y_test, y_pred_best)
            
            st.markdown(f"""
            **📊 Regression Metrics Explained:**
            - **R² Score: {r2:.3f}** - Proportion of variance explained (1.0 = perfect fit)
            - **RMSE: {np.sqrt(mse):.2f}** - Average prediction error in original units
            - **MAE: {mae:.2f}** - Mean absolute error (robust to outliers)
            """)
            
            # 2D Visualization
            if len(feature_names) >= 2:
                st.subheader("📊 2D Feature Visualization")
                
                col1, col2 = st.columns(2)
                with col1:
                    feature_x = st.selectbox("Choose X-axis feature:", feature_names, key="reg_2d_x")
                with col2:
                    feature_y = st.selectbox("Choose Y-axis feature:", 
                                           [f for f in feature_names if f != feature_x], key="reg_2d_y")
                
                if len(feature_names) == 2:
                    fig = create_regression_surface_plot(X_train, y_train, best_model, feature_x, feature_y, best_model_name, "Survived")
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 **Prediction Surface**: The colored surface shows the model's predictions across the feature space.")
                else:
                    fig = create_2d_scatter_plot(X_test, y_test, y_pred_best, feature_x, feature_y, "Regression", "Survived")
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 **2D Feature Plot**: Each point is a passenger. Color intensity shows the target value. Size shows prediction accuracy.")
            
            # Feature Importance
            if hasattr(best_model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                           title=f'Which Features Matter Most? ({best_model_name})',
                           color_discrete_sequence=['lightgreen'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 **Feature Importance** shows which passenger characteristics the {best_model_name} considers most important.")
            
            elif hasattr(best_model, 'coef_'):
                st.subheader("🎯 Feature Coefficients")
                
                feature_coef = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': best_model.coef_
                }).sort_values('coefficient', ascending=True)
                
                fig = px.bar(feature_coef, x='coefficient', y='feature', orientation='h',
                           title=f'Feature Coefficients ({best_model_name})',
                           color='coefficient', color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 **Feature Coefficients** show how much each feature influences the {best_model_name}'s predictions.")
            
            # Interactive Prediction
            st.markdown('<h2 class="section-header">🔮 Make Your Own Predictions</h2>', unsafe_allow_html=True)
            
            st.markdown("Try different passenger profiles and see what the AI predicts!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Passenger Profile")
                
                prediction_inputs = {}
                
                if 'Age' in selected_features:
                    prediction_inputs['age'] = st.slider("Age", 0, 80, 30, key="reg_pred_age")
                if 'Sex' in selected_features:
                    sex_input = st.selectbox("Gender", ["Female", "Male"], key="reg_pred_sex")
                    prediction_inputs['sex_encoded'] = 1 if sex_input == "Male" else 0
                if 'Passenger Class' in selected_features:
                    prediction_inputs['pclass'] = st.selectbox("Passenger Class", [1, 2, 3], index=1, key="reg_pred_pclass")
                if 'Fare' in selected_features:
                    prediction_inputs['fare'] = st.slider("Fare ($)", 0, 500, 50, key="reg_pred_fare")
                if 'Siblings/Spouses' in selected_features:
                    prediction_inputs['sibsp'] = st.slider("Siblings/Spouses Aboard", 0, 8, 0, key="reg_pred_sibsp")
                if 'Parents/Children' in selected_features:
                    prediction_inputs['parch'] = st.slider("Parents/Children Aboard", 0, 6, 0, key="reg_pred_parch")
                if 'Port of Embarkation' in selected_features:
                    port_input = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"], key="reg_pred_port")
                    port_mapping = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
                    prediction_inputs['embarked_encoded'] = port_mapping[port_input]
            
            with col2:
                st.subheader("🎯 Prediction Result")
                
                if st.button("🔮 Predict Value", type="primary", key="reg_predict_btn"):
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
    
    # =============================================
    # PAGE 3: CLASSIFICATION
    # =============================================
    elif page == "🎯 Classification":
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
        selected_features = st.session_state.get('selected_features', [])
        
        st.markdown('<h2 class="section-header">🤖 Classification Models Training & Comparison</h2>', unsafe_allow_html=True)
        
        # Model selection in page
        st.markdown("### 📋 Model Selection")
        
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
        
        selected_models = st.multiselect(
            "Choose Classification Models:",
            list(available_models.keys()),
            default=["Random Forest", "Logistic Regression", "Support Vector Machine"],
            help="Select multiple models to compare their performance",
            key="classification_models"
        )
        
        # Get test size from session state
        test_size = st.session_state.get("test_size", 20)
        
        # Hyperparameters in page
        st.markdown("### ⚙️ Hyperparameter Tuning")
        st.info("💡 Hyperparameters control how models learn. Experiment with different values!")
        hyperparams = {}
        
        for model_name in selected_models:
            with st.expander(f"🔧 {model_name} - Hyperparameters"):
                if model_name == "Logistic Regression":
                    hyperparams[model_name] = {
                        'C': st.slider(f"Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"clf_lr_c"),
                        'max_iter': st.slider(f"Max Iterations", 100, 2000, 1000, 100, key=f"clf_lr_iter"),
                        'solver': st.selectbox(f"Solver", ['liblinear', 'lbfgs', 'saga'], index=1, key=f"clf_lr_solver")
                    }
                    
                elif model_name == "Random Forest":
                    hyperparams[model_name] = {
                        'n_estimators': st.slider(f"Number of Trees", 10, 500, 100, 10, key=f"clf_rf_trees"),
                        'max_depth': st.slider(f"Max Depth", 3, 20, 10, 1, key=f"clf_rf_depth"),
                        'min_samples_split': st.slider(f"Min Samples Split", 2, 20, 2, 1, key=f"clf_rf_split"),
                        'min_samples_leaf': st.slider(f"Min Samples Leaf", 1, 10, 1, 1, key=f"clf_rf_leaf")
                    }
                    
                elif model_name == "Support Vector Machine":
                    hyperparams[model_name] = {
                        'C': st.slider(f"Regularization (C)", 0.01, 10.0, 1.0, 0.01, key=f"clf_svm_c"),
                        'kernel': st.selectbox(f"Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], key=f"clf_svm_kernel"),
                        'gamma': st.selectbox(f"Gamma", ['scale', 'auto'], key=f"clf_svm_gamma")
                    }
                    
                elif model_name == "Decision Tree":
                    hyperparams[model_name] = {
                        'max_depth': st.slider(f"Max Depth", 3, 20, 10, 1, key=f"clf_dt_depth"),
                        'min_samples_split': st.slider(f"Min Samples Split", 2, 20, 2, 1, key=f"clf_dt_split"),
                        'min_samples_leaf': st.slider(f"Min Samples Leaf", 1, 10, 1, 1, key=f"clf_dt_leaf"),
                        'criterion': st.selectbox(f"Criterion", ['gini', 'entropy'], key=f"clf_dt_criterion")
                    }
                    
                elif model_name == "K-Nearest Neighbors":
                    hyperparams[model_name] = {
                        'n_neighbors': st.slider(f"Number of Neighbors", 1, 20, 5, 1, key=f"clf_knn_n"),
                        'weights': st.selectbox(f"Weights", ['uniform', 'distance'], key=f"clf_knn_weights"),
                        'metric': st.selectbox(f"Distance Metric", ['euclidean', 'manhattan', 'minkowski'], key=f"clf_knn_metric")
                    }
                    
                elif model_name == "Gradient Boosting":
                    hyperparams[model_name] = {
                        'n_estimators': st.slider(f"Number of Estimators", 50, 300, 100, 10, key=f"clf_gb_trees"),
                        'learning_rate': st.slider(f"Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"clf_gb_lr"),
                        'max_depth': st.slider(f"Max Depth", 3, 10, 6, 1, key=f"clf_gb_depth")
                    }
                    
                elif model_name == "AdaBoost":
                    hyperparams[model_name] = {
                        'n_estimators': st.slider(f"Number of Estimators", 10, 200, 50, 10, key=f"clf_ada_trees"),
                        'learning_rate': st.slider(f"Learning Rate", 0.1, 2.0, 1.0, 0.1, key=f"clf_ada_lr"),
                        'algorithm': st.selectbox(f"Algorithm", ['SAMME', 'SAMME.R'], key=f"clf_ada_algo")
                    }
                    
                elif model_name == "Naive Bayes":
                    hyperparams[model_name] = {
                        'var_smoothing': st.slider(f"Smoothing", 1e-10, 1e-5, 1e-9, 1e-10, key=f"clf_nb_smooth", format="%.2e")
                    }
                    
                elif model_name == "Neural Network":
                    layer_sizes = st.multiselect(f"Hidden Layer Sizes", [50, 100, 200, 300], default=[100], key=f"clf_nn_layers")
                    hyperparams[model_name] = {
                        'hidden_layer_sizes': tuple(layer_sizes) if layer_sizes else (100,),
                        'learning_rate_init': st.slider(f"Learning Rate", 0.001, 0.1, 0.001, 0.001, key=f"clf_nn_lr"),
                        'max_iter': st.slider(f"Max Iterations", 100, 1000, 200, 50, key=f"clf_nn_iter"),
                        'activation': st.selectbox(f"Activation", ['relu', 'tanh', 'logistic'], key=f"clf_nn_activation")
                    }
                    
                elif model_name == "Ridge Classifier":
                    hyperparams[model_name] = {
                        'alpha': st.slider(f"Alpha (Regularization)", 0.1, 10.0, 1.0, 0.1, key=f"clf_ridge_alpha"),
                    }
        
        if selected_models:
            # Train models
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y
            )
            
            models = {}
            results = []
            
            for model_name in selected_models:
                params = hyperparams.get(model_name, {})
                
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
                        random_state=42
                    )
                
                # Train model
                model.fit(X_train, y_train)
                models[model_name] = model
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
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
            
            # Display results
            results_df = pd.DataFrame(results)
            
            st.subheader("📊 Model Performance Comparison")
            
            # Metrics display
            for result in results:
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
            
            # Best model
            best_model_name = results_df.loc[results_df['Test Score'].idxmax(), 'Model']
            best_model = models[best_model_name]
            y_pred_best = best_model.predict(X_test)
            
            st.subheader(f"🏆 Best Model: {best_model_name}")
            
            # Confusion Matrix
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
                    feature_x = st.selectbox("Choose X-axis feature:", feature_names, key="clf_2d_x")
                with col2:
                    feature_y = st.selectbox("Choose Y-axis feature:", 
                                           [f for f in feature_names if f != feature_x], key="clf_2d_y")
                
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
                
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                           title=f'Which Features Matter Most? ({best_model_name})',
                           color_discrete_sequence=['lightgreen'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 **Feature Importance** shows which passenger characteristics the {best_model_name} considers most important for predicting survival.")
            
            elif hasattr(best_model, 'coef_'):
                st.subheader("🎯 Feature Coefficients")
                
                feature_coef = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': best_model.coef_[0] if len(best_model.coef_.shape) > 1 else best_model.coef_
                }).sort_values('coefficient', ascending=True)
                
                fig = px.bar(feature_coef, x='coefficient', y='feature', orientation='h',
                           title=f'Feature Coefficients ({best_model_name})',
                           color='coefficient', color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 **Feature Coefficients** show how much each feature influences the {best_model_name}'s predictions. Positive values increase survival probability, negative values decrease it.")
            
            # Interactive Prediction
            st.markdown('<h2 class="section-header">🔮 Make Your Own Predictions</h2>', unsafe_allow_html=True)
            
            st.markdown("Try different passenger profiles and see what the AI predicts!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Passenger Profile")
                
                prediction_inputs = {}
                
                if 'Age' in selected_features:
                    prediction_inputs['age'] = st.slider("Age", 0, 80, 30, key="clf_pred_age")
                if 'Sex' in selected_features:
                    sex_input = st.selectbox("Gender", ["Female", "Male"], key="clf_pred_sex")
                    prediction_inputs['sex_encoded'] = 1 if sex_input == "Male" else 0
                if 'Passenger Class' in selected_features:
                    prediction_inputs['pclass'] = st.selectbox("Passenger Class", [1, 2, 3], index=1, key="clf_pred_pclass")
                if 'Fare' in selected_features:
                    prediction_inputs['fare'] = st.slider("Fare ($)", 0, 500, 50, key="clf_pred_fare")
                if 'Siblings/Spouses' in selected_features:
                    prediction_inputs['sibsp'] = st.slider("Siblings/Spouses Aboard", 0, 8, 0, key="clf_pred_sibsp")
                if 'Parents/Children' in selected_features:
                    prediction_inputs['parch'] = st.slider("Parents/Children Aboard", 0, 6, 0, key="clf_pred_parch")
                if 'Port of Embarkation' in selected_features:
                    port_input = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"], key="clf_pred_port")
                    port_mapping = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
                    prediction_inputs['embarked_encoded'] = port_mapping[port_input]
            
            with col2:
                st.subheader("🎯 Prediction Result")
                
                if st.button("🔮 Predict Survival", type="primary", key="clf_predict_btn"):
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

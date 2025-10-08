"""
Preprocessing module for the ML Demo application.
Handles data preprocessing, encoding, and feature preparation.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def get_feature_mapping():
    """
    Get the mapping from user-friendly feature names to dataframe column names.
    
    Returns:
        dict: Mapping of display names to column names
    """
    return {
        'Age': 'age',
        'Sex': 'sex_encoded',
        'Passenger Class': 'pclass',
        'Fare': 'fare',
        'Siblings/Spouses': 'sibsp',
        'Parents/Children': 'parch',
        'Port of Embarkation': 'embarked_encoded'
    }


def encode_categorical_features(df):
    """
    Encode categorical features in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    df_encoded = df.copy()
    df_encoded['sex_encoded'] = (df_encoded['sex'] == 'male').astype(int)
    df_encoded['embarked_encoded'] = LabelEncoder().fit_transform(df_encoded['embarked'].fillna('S'))
    return df_encoded


def preprocess_data(df, selected_features, missing_age_option, normalize):
    """
    Preprocess the data: handle missing values, encode features, and normalize.
    
    Args:
        df (pd.DataFrame): Input dataframe
        selected_features (list): List of selected feature names
        missing_age_option (str): How to handle missing ages
        normalize (bool): Whether to normalize features
        
    Returns:
        tuple: (X, y, feature_names, scaler, before_count, after_count)
    """
    processed_df = df.copy()
    
    # Handle missing ages
    if missing_age_option == "Fill with median":
        processed_df['age'] = processed_df['age'].fillna(processed_df['age'].median())
    elif missing_age_option == "Fill with mean":
        processed_df['age'] = processed_df['age'].fillna(processed_df['age'].mean())
    
    # Encode categorical variables
    processed_df = encode_categorical_features(processed_df)
    
    # Feature mapping
    feature_mapping = get_feature_mapping()
    feature_names = [feature_mapping[f] for f in selected_features]
    X = processed_df[feature_names].copy()
    
    # Drop rows with missing values
    before_count = len(X)
    X = X.dropna()
    y = processed_df.loc[X.index, 'survived']
    after_count = len(X)
    
    # Normalize if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X, y, feature_names, scaler, before_count, after_count


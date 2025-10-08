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
        'Port of Embarkation': 'embarked_encoded',
        'Survived': 'survived'
    }


def get_reverse_feature_mapping():
    """
    Get the reverse mapping from dataframe column names to user-friendly feature names.
    Handles multiple possible column names that could map to the same feature.
    
    Returns:
        dict: Mapping of column names to display names
    """
    reverse_map = {}
    feature_mapping = get_feature_mapping()
    
    for feature_name, column_name in feature_mapping.items():
        reverse_map[column_name] = feature_name
    
    # Add additional mappings for raw column names that might be used as target
    reverse_map['sex'] = 'Sex'
    reverse_map['embarked'] = 'Port of Embarkation'
    
    return reverse_map


def get_categorical_feature_names():
    """
    Get the list of user-friendly feature names that are categorical.
    
    Returns:
        list: List of categorical feature names
    """
    return ['Sex', 'Port of Embarkation']


def get_categorical_columns(df):
    """
    Identify categorical columns in dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        list: List of categorical column names
    """
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
    return categorical_cols


def encode_categorical_features(df, method='Drop Columns', exclude_cols=None):
    """
    Encode categorical features based on method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): 'Drop Columns', 'Label Encoding', or 'One-Hot Encoding'
        exclude_cols (list): Columns to exclude from encoding (e.g., target)
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    df_encoded = df.copy()
    exclude_cols = exclude_cols or []
    
    # Get categorical columns
    categorical_cols = get_categorical_columns(df_encoded)
    
    # Exclude specific columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    if method == 'Drop Columns':
        # Drop categorical columns except already encoded ones
        # But keep sex and embarked for backward compatibility
        df_encoded['sex_encoded'] = (df_encoded['sex'] == 'male').astype(int) if 'sex' in df_encoded.columns else 0
        if 'embarked' in df_encoded.columns:
            df_encoded['embarked_encoded'] = LabelEncoder().fit_transform(df_encoded['embarked'].fillna('S'))
        # Don't drop sex and embarked yet - will be handled by feature selection
        
    elif method == 'Label Encoding':
        # Encode all categorical columns with LabelEncoder
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle NaN values
            non_null_mask = df_encoded[col].notna()
            if non_null_mask.any():
                df_encoded.loc[non_null_mask, f'{col}_encoded'] = le.fit_transform(df_encoded.loc[non_null_mask, col].astype(str))
            else:
                df_encoded[f'{col}_encoded'] = 0
        
        # Also handle sex and embarked for backward compatibility
        if 'sex' in df_encoded.columns:
            df_encoded['sex_encoded'] = (df_encoded['sex'] == 'male').astype(int)
        if 'embarked' in df_encoded.columns:
            df_encoded['embarked_encoded'] = LabelEncoder().fit_transform(df_encoded['embarked'].fillna('S'))
    
    elif method == 'One-Hot Encoding':
        # Use pandas get_dummies for one-hot encoding
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Get dummies and add to dataframe
                dummies = pd.get_dummies(df_encoded[col], prefix=col, prefix_sep='_', drop_first=False, dummy_na=False)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Also handle sex and embarked for backward compatibility
        if 'sex' in df_encoded.columns:
            df_encoded['sex_encoded'] = (df_encoded['sex'] == 'male').astype(int)
        if 'embarked' in df_encoded.columns:
            df_encoded['embarked_encoded'] = LabelEncoder().fit_transform(df_encoded['embarked'].fillna('S'))
    
    return df_encoded


def preprocess_data(df, selected_features, missing_age_option, normalize, 
                   target_column='survived', encoding_method='Drop Columns'):
    """
    Preprocess the data: handle missing values, encode features, and normalize.
    
    Args:
        df (pd.DataFrame): Input dataframe
        selected_features (list): List of selected feature names
        missing_age_option (str): How to handle missing ages
        normalize (bool): Whether to normalize features
        target_column (str): Name of the target column to predict
        encoding_method (str): Method for encoding categorical features
        
    Returns:
        tuple: (X, y, feature_names, scaler, before_count, after_count, df_transformed)
    """
    processed_df = df.copy()
    
    # Handle missing ages
    if missing_age_option == "Fill with median":
        processed_df['age'] = processed_df['age'].fillna(processed_df['age'].median())
    elif missing_age_option == "Fill with mean":
        processed_df['age'] = processed_df['age'].fillna(processed_df['age'].mean())
    
    # Encode categorical variables (exclude target column)
    processed_df = encode_categorical_features(processed_df, method=encoding_method, exclude_cols=[target_column])
    
    # Feature mapping
    feature_mapping = get_feature_mapping()
    
    # Build feature_names list based on encoding method
    feature_names = []
    for f in selected_features:
        mapped_col = feature_mapping[f]
        
        # For One-Hot Encoding, replace encoded columns with all one-hot columns
        if encoding_method == 'One-Hot Encoding':
            # Check if this is a categorical feature that was one-hot encoded
            if mapped_col == 'sex_encoded':
                # Find all sex_* columns (excluding sex_encoded)
                sex_cols = [col for col in processed_df.columns if col.startswith('sex_') and col != 'sex_encoded']
                if sex_cols:
                    feature_names.extend(sex_cols)
                else:
                    feature_names.append(mapped_col)
            elif mapped_col == 'embarked_encoded':
                # Find all embarked_* columns (excluding embarked_encoded)
                embarked_cols = [col for col in processed_df.columns if col.startswith('embarked_') and col != 'embarked_encoded']
                if embarked_cols:
                    feature_names.extend(embarked_cols)
                else:
                    feature_names.append(mapped_col)
            else:
                feature_names.append(mapped_col)
        else:
            # For Label Encoding and Drop Columns, use the mapped column directly
            feature_names.append(mapped_col)
    
    X = processed_df[feature_names].copy()
    
    # Drop rows with missing values
    before_count = len(X)
    X = X.dropna()
    
    # Extract target variable
    y = processed_df.loc[X.index, target_column]
    
    # If target is categorical, encode it
    if pd.api.types.is_categorical_dtype(y) or y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=target_column)
    
    after_count = len(X)
    
    # Normalize if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Create transformed dataframe for display (after normalization)
    df_transformed = X.copy()
    df_transformed[target_column] = y
    
    return X, y, feature_names, scaler, before_count, after_count, df_transformed


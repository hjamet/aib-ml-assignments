"""
Data loading module for the ML Demo application.
Handles loading and caching of the Titanic dataset.
"""

import streamlit as st
import seaborn as sns


@st.cache_data
def load_titanic_data():
    """
    Load and prepare the Titanic dataset.
    
    Returns:
        pd.DataFrame: The Titanic dataset
    """
    df = sns.load_dataset('titanic')
    return df


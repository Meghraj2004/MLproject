import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load the student performance dataset."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data, target_variable=None, categorical_encoding='label', test_size=0.2, random_state=42):
    """
    Preprocess the data for machine learning models.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    target_variable : str, optional
        The target variable for supervised learning
    categorical_encoding : str, default='label'
        Method for encoding categorical variables ('label' or 'onehot')
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing preprocessed data and related information
    """
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Handle missing values if any
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if target_variable and target_variable in numerical_cols:
        numerical_cols.remove(target_variable)
    if target_variable and target_variable in categorical_cols:
        categorical_cols.remove(target_variable)
    
    # Create result dictionary
    result = {
        'original_data': data,
        'processed_data': df,
        'categorical_columns': categorical_cols,
        'numerical_columns': numerical_cols,
        'encoding_method': categorical_encoding
    }
    
    # For supervised learning
    if target_variable:
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        
        # Encode categorical features
        if categorical_encoding == 'label':
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            result['label_encoders'] = label_encoders
        
        elif categorical_encoding == 'onehot':
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols_X = [col for col in numerical_cols if col in X.columns]
        if numerical_cols_X:
            X_train[numerical_cols_X] = scaler.fit_transform(X_train[numerical_cols_X])
            X_test[numerical_cols_X] = scaler.transform(X_test[numerical_cols_X])
        
        # Encode target if it's categorical
        if target_variable in data.select_dtypes(include=['object', 'category']).columns:
            le_target = LabelEncoder()
            y_train = le_target.fit_transform(y_train)
            y_test = le_target.transform(y_test)
            result['target_encoder'] = le_target
        
        result.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'target_variable': target_variable
        })
    
    return result

def create_target_from_scores(data, threshold=60):
    """Create a binary target variable based on average test scores."""
    df = data.copy()
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['pass_fail'] = (df['average_score'] >= threshold).astype(int)
    return df

def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def calculate_classification_metrics(y_true, y_pred):
    """Calculate classification evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def get_feature_importance(model, feature_names):
    """Extract feature importance from a model if available."""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) == 1:
            return dict(zip(feature_names, model.coef_))
        else:
            return dict(zip(feature_names, model.coef_[0]))
    else:
        return None

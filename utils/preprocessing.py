import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def remove_outliers(df, columns, n_std=3):
    """Remove outliers based on standard deviation"""
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[abs(df[col] - mean) <= (n_std * std)]
    return df

def create_features(df):
    """Create additional features that might be useful for prediction"""
    df = df.copy()
    
    # Age of the house
    current_year = pd.Timestamp.now().year
    df['house_age'] = current_year - df['year_built']
    
    # Price per square foot
    if 'price' in df.columns:
        df['price_per_sqft'] = df['price'] / df['area']
    
    # Room ratios
    df['bedroom_ratio'] = df['bedrooms'] / df['area']
    df['bathroom_ratio'] = df['bathrooms'] / df['area']
    
    # Total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    return df

def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Compute house_age as (2024 - year_built) so that newer houses have higher values
    df['house_age'] = 2024 - df['year_built']
    
    # Create squared terms to emphasize more bedrooms and bathrooms
    df['bedrooms_squared'] = df['bedrooms'] ** 2  # Emphasize more bedrooms
    df['bathrooms_squared'] = df['bathrooms'] ** 2  # Emphasize more bathrooms
    df['area_squared'] = df['area'] ** 2  # Emphasize larger area
    
    # Compute total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Remove outliers using IQR method
    numeric_cols = ['area', 'bedrooms', 'bathrooms', 'year_built', 'price']
    df = remove_outliers(df, numeric_cols)
    
    # Scale features
    scaler = StandardScaler()
    features_to_scale = [
        'area', 'bedrooms', 'bathrooms', 'house_age',
        'bedrooms_squared', 'bathrooms_squared', 'area_squared',
        'total_rooms'
    ]
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Save the scaler for later use in prediction
    joblib.dump(scaler, 'model/scaler.joblib')
    
    # Only return the features needed for the model and the target
    return df[features_to_scale + ['price']]

def preprocess_data_predict(input_data):
    # Create a copy to avoid modifying the original dataframe
    df = input_data.copy()
    
    # Compute house_age as (2024 - year_built) so that newer houses have higher values
    df['house_age'] = 2024 - df['year_built']
    
    # Create squared terms to emphasize more bedrooms and bathrooms
    df['bedrooms_squared'] = df['bedrooms'] ** 2  # Emphasize more bedrooms
    df['bathrooms_squared'] = df['bathrooms'] ** 2  # Emphasize more bathrooms
    df['area_squared'] = df['area'] ** 2  # Emphasize larger area
    
    # Compute total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Load the scaler and scale features
    scaler = joblib.load('model/scaler.joblib')
    features_to_scale = [
        'area', 'bedrooms', 'bathrooms', 'house_age',
        'bedrooms_squared', 'bathrooms_squared', 'area_squared',
        'total_rooms'
    ]
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    
    # Only return the features needed for the model
    return df[features_to_scale]

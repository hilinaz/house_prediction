import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

def preprocess_data(data):
    """Preprocess data for training"""
    # Create features
    data = create_features(data)
    
    # Remove outliers
    numeric_cols = ['area', 'bedrooms', 'bathrooms', 'year_built', 'price']
    data = remove_outliers(data, numeric_cols)
    
    # Define features and target
    features = ['area', 'bedrooms', 'bathrooms', 'year_built', 
                'house_age', 'bedroom_ratio', 'bathroom_ratio', 'total_rooms']
    target = 'price'
    
    X = data[features]
    y = data[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, 'model/scaler.pkl')
    
    return pd.DataFrame(X_scaled, columns=features), y

def preprocess_data_predict(data):
    """Preprocess data for prediction"""
    # Create features
    data = create_features(data)
    
    # Define features
    features = ['area', 'bedrooms', 'bathrooms', 'year_built', 
                'house_age', 'bedroom_ratio', 'bathroom_ratio', 'total_rooms']
    
    X = data[features]
    
    # Load and apply scaler
    import joblib
    scaler = joblib.load('model/scaler.pkl')
    X_scaled = scaler.transform(X)
    
    return pd.DataFrame(X_scaled, columns=features)

def preprocess_data(data):
    # Select features and target
    features = ['area', 'bedrooms', 'bathrooms', 'year_built']
    target = 'price'
    X = data[features]
    y = data[target]
    return X, y

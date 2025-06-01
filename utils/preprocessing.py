def preprocess_data(data):
    # Used only during training where 'price' column exists
    features = ['area', 'bedrooms', 'bathrooms', 'year_built']
    target = 'price'
    X = data[features]
    y = data[target]
    return X, y

def preprocess_data_predict(data):
    # Used during prediction where 'price' is not available
    features = ['area', 'bedrooms', 'bathrooms', 'year_built']
    X = data[features]
    return X

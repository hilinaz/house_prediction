import joblib
import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_data_predict
import sys
import os

# Allow running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def predict_price(input_data):
    """
    Predict house price based on input features.
    Ensures predictions are non-negative and reasonable.
    """
    # Load model
    model = joblib.load('model/model.pkl')
    
    # Preprocess input data
    X = preprocess_data_predict(input_data)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Ensure prediction is non-negative
    prediction = max(0, prediction)
    
    return prediction

if __name__ == "__main__":
    # Prompt user for input
    try:
        print("Enter house features for price prediction:")
        area = float(input("Area (in sqft): "))
        bedrooms = int(input("Number of bedrooms: "))
        bathrooms = int(input("Number of bathrooms: "))
        year_built = int(input("Year built: "))
        
        new_data = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'year_built': year_built
        }])
        
        prediction = predict_price(new_data)
        print(f"\n🏠 Input Features:")
        for col, val in new_data.iloc[0].items():
            print(f"{col}: {val}")
        print(f"\n💰 Predicted Price: ${prediction:,.2f}")
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")

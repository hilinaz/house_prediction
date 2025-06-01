import joblib
import pandas as pd

# Load model
model = joblib.load('model/model.pkl')


# Example input
new_data = pd.DataFrame([{
    'area': 1800,
    'bedrooms': 3,
    'bathrooms': 2,
    'year_built': 2015
}])

# Predict
prediction = model.predict(new_data)
print(f"💰 Predicted Price: ${prediction[0]:,.2f}")

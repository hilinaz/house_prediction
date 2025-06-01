from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('../model/model.pkl')

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict_price(area: float, bedrooms: int, bathrooms: int, year_built: int):
    input_df = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built
    }])
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(prediction, 2)}

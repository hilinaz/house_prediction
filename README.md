# 🏠 House Price Prediction

Estimate house prices based on features like area, number of bedrooms/bathrooms, and year built.

## 🔧 Project Structure

```
house-price-prediction/
├── data/                # Dataset
├── model/               # Training and prediction code
├── app/                 # Optional FastAPI app
├── utils/               # Preprocessing helpers
├── report/              # Technical Report
└── README.md            # Project overview
```

## 🚀 How to Run

1. Install requirements: `pip install pandas scikit-learn joblib fastapi uvicorn`
2. Train model: `python model/train_model.py`
3. Predict: `python model/predict.py`
4. Run API: `uvicorn app.app:app --reload`

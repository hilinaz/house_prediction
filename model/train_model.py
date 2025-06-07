import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from utils.preprocessing import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("📊 Loading data...")
data = pd.read_csv('data/house_data.csv')

# Preprocess data
print("🔄 Preprocessing data...")
X, y = preprocess_data(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model
print("🤖 Training model...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',
    early_stopping_rounds=50
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Performance Metrics:")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('report/feature_importance.png')

# Save model
joblib.dump(model, 'model/model.pkl')
print("\n✅ Model trained and saved to model/model.pkl")

# Save metrics to file
with open('report/model_metrics.txt', 'w') as f:
    f.write(f"Mean Absolute Error: ${mae:,.2f}\n")
    f.write(f"Root Mean Squared Error: ${rmse:,.2f}\n")
    f.write(f"R² Score: {r2:.4f}\n")

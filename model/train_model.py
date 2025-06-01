import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from utils.preprocessing import preprocess_data

# Load data
data = pd.read_csv('data/house_data.csv')
X, y = preprocess_data(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/model.pkl')
print("✅ Model trained and saved to model/model.pkl")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset.csv')

# Drop missing values for simplicity
df.dropna(inplace=True)

# Feature and target selection
X = df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']]
y = df['AQI']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Feature Importance
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()
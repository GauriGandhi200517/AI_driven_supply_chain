import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('supply_chain_data.csv')

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data.fillna(method='ffill')

# Features and target
X = data.drop('Demand', axis=1)
y = data['Demand']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model
import joblib
joblib.dump(model, 'demand_forecasting_model.pkl')

print("Model saved as 'demand_forecasting_model.pkl'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
house_data = pd.read_csv('house_prices.csv')

# Remove outliers
z_scores = (house_data['price'] - house_data['price'].mean()) / house_data['price'].std()
house_data = house_data[(np.abs(z_scores) < 3)]  # Remove rows with z-scores > 3

# One-hot encode the 'location' column
house_data = pd.get_dummies(house_data, columns=['location'], drop_first=True)

# Normalize the target variable (price)
scaler_target = MinMaxScaler()
house_data['price'] = scaler_target.fit_transform(house_data[['price']])

# Define features and target
X = house_data.drop(columns=['price'])
y = house_data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler_features = StandardScaler()
X_train = scaler_features.fit_transform(X_train)
X_test = scaler_features.transform(X_test)

# Gradient Boosting Regressor with hyperparameter tuning
gbr = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error',
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_gbr = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the best model
y_train_pred = best_gbr.predict(X_train)
y_test_pred = best_gbr.predict(X_test)

train_score = best_gbr.score(X_train, y_train)
test_score = best_gbr.score(X_test, y_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Print scores and metrics
print(f"Training R-squared: {train_score}")
print(f"Testing R-squared: {test_score}")
print(f"Training MAE: {mae_train}")
print(f"Testing MAE: {mae_test}")
print(f"Training MSE: {mse_train}")
print(f"Testing MSE: {mse_test}")

# Rescale metrics to original scale
mae_train_original = scaler_target.inverse_transform([[mae_train]])[0][0]
mae_test_original = scaler_target.inverse_transform([[mae_test]])[0][0]

print(f"Training MAE (original scale): {mae_train_original}")
print(f"Testing MAE (original scale): {mae_test_original}")

# Plotting actual vs predicted values
plt.figure(figsize=(14, 6))

# Scatter plot for actual vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, label='Test Data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red',
         linestyle='--', label='Ideal Fit')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices (Normalized)')
plt.ylabel('Predicted Prices (Normalized)')
plt.legend()

# Bar plot for scores and metrics
plt.subplot(1, 2, 2)
metrics = ['R-squared (Train)', 'R-squared (Test)', 'MAE (Train)', 'MAE (Test)', 'MSE (Train)', 'MSE (Test)']
values = [train_score, test_score, mae_train, mae_test, mse_train, mse_test]
plt.bar(metrics, values, color=['blue', 'blue', 'green', 'green', 'orange', 'orange'])
plt.title('Model Evaluation Metrics')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

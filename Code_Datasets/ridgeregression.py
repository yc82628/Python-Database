# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'employee_salary_data.csv' with your actual dataset)
data = pd.read_csv('employee_salary_data.csv')

# Check the first few rows to understand the structure of the dataset
print(data.head())

# One-hot encode categorical features (if applicable, such as 'JobTitle', 'Department', etc.)
# Modify this based on actual categorical columns in your dataset
data = pd.get_dummies(data, drop_first=True)

# Assuming 'Salary' is the target variable and the rest are features
X = data.drop('Salary', axis=1)  # Features (excluding 'Salary')
y = data['Salary']  # Target variable (Salary)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data (important for Ridge regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Ridge regression model
ridge_regressor = Ridge(alpha=0.5)  # alpha is the regularization parameter

# Train the model
ridge_regressor.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = ridge_regressor.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')




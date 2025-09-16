import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
data = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=" ", skipinitialspace=True)

# Handle missing values
data.dropna(inplace=True)

# One-hot encode 'origin' feature
data = pd.get_dummies(data, columns=['origin'], drop_first=False)

# Define features and target
X = data.drop('mpg', axis=1)
y = data['mpg']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train MAE:", mean_absolute_error(y_train, y_train_pred))
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))

print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))

# Feature importance
print("Feature Coefficients:", model.coef_)

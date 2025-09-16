import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# Replace 'your_dataset.csv' with your actual dataset file
data = pd.read_csv("house_prices.csv")

# One-hot encode categorical variables (e.g., location)
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Separate features (X) and target (y)
X = data.drop('price', axis=1) #Excluding the Price features.
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Elastic Net model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# Train the model
elastic_net.fit(X_train_scaled, y_train)

# Make predictions
y_pred = elastic_net.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot True vs. Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("True vs. Predicted House Prices")
plt.grid()
plt.show()

# Predict a new sample (example)
# Replace the values with actual inputs for a house
new_house = [[1360, 3, 1, 1]]  # Example: [size, bedrooms, pool, location_encoded_1, location_encoded_2]
new_house_scaled = scaler.transform(new_house)
predicted_price = elastic_net.predict(new_house_scaled)
print("Predicted House Price:", y_pred)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
house_data = pd.read_csv('house_prices.csv')

# One-hot encode the 'location' column
house_data = pd.get_dummies(house_data, columns=['location'], drop_first=True)

# Convert 'price' into a binary classification (e.g., 'Affordable' vs 'Expensive')
threshold = 500000  # Set your threshold for affordability
house_data['price_category'] = (house_data['price'] > threshold).astype(int)

# Define features and target
X = house_data[['size', 'bedrooms', 'location_B', 'location_C']]
y = house_data['price_category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Best-fit plot (for a single feature, e.g., 'size')
feature = 'size'
X_single = X[feature].values.reshape(-1, 1)  # Single feature for visualization
model_single = LogisticRegression()         # Train on the single feature
model_single.fit(X_single, y)

# Generate predicted probabilities for the feature range
X_range = np.linspace(X_single.min(), X_single.max(), 300).reshape(-1, 1)
y_prob = model_single.predict_proba(X_range)[:, 1]  # Probability of class 1

# Plot the data points and the sigmoid curve
plt.figure(figsize=(8, 6))
plt.scatter(X[feature], y, color='blue', label='Data Points')
plt.plot(X_range, y_prob, color='red', linewidth=2, label='Logistic Fit')
plt.xlabel(feature)
plt.ylabel('Probability of Expensive (1)')
plt.title('Logistic Regression Best-Fit Curve')
plt.legend()
plt.grid()
plt.show()

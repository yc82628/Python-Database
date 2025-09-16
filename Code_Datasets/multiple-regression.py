import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
house_data = pd.read_csv('house_prices.csv')

# One-hot encode the 'location' column
house_data = pd.get_dummies(house_data, columns=['location'], drop_first=True)

# Define features and target
X = house_data[['size', 'bedrooms', 'location_B', 'location_C']]
y = house_data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42))

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f'R-squared: {score}')

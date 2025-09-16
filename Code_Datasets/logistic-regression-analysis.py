import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'heart_disease.csv'
data = pd.read_csv(file_path)

# Data preprocessing
# Encode categorical variables
categorical_cols = ['sex', 'cp', 'restecg', 'thal', 'slope']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Treat 'num' as binary: 0 (no disease) and 1 (any disease)
data_encoded['num'] = data_encoded['num'].apply(lambda x: 1 if x > 0 else 0)

# Define features (X) and target (y)
X = data_encoded.drop(columns=['id', 'dataset', 'num'])
y = data_encoded['num']

# Handle missing values
# Impute missing values for numerical columns with their mean
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    X[col].fillna(X[col].mean(), inplace=True)

# Ensure consistency between X and y
X = X.dropna()
y = y.loc[X.index]  # Update y to match X after dropping rows

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation: Classification report and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Feature importance visualization
coefficients = pd.Series(model.coef_[0], index=X.columns)
coefficients = coefficients.sort_values()

plt.figure(figsize=(10, 8))
coefficients.plot(kind='barh', color='teal')
plt.title('Feature Importance')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

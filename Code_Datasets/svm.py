import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'heart_disease.csv'  # Replace with the path to your file
data = pd.read_csv(file_path)

# Step 1: Preprocessing
# Select relevant columns: 'age', 'chol' as features and 'num' as target
columns_to_use = ['age', 'chol', 'num']
df = data[columns_to_use].copy()

# Handle missing values by imputing the mean
imputer = SimpleImputer(strategy='mean')
df[['age', 'chol']] = imputer.fit_transform(df[['age', 'chol']])

# Binary classification: Convert target 'num' to binary (0: no disease, 1: disease)
df['num'] = (df['num'] > 0).astype(int)

# Step 2: Train-test split
X = df[['age', 'chol']].values
y = df['num'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train SVM with RBF kernel
svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1, gamma=0.5, probability=True))
])
svm_model.fit(X_train, y_train)

# Step 4: Generate Classification Report and Accuracy Score
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
print(f"Accuracy: {accuracy:.2f}")

# Extract metrics from the classification report for plotting
report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"], output_dict=True)
categories = list(report.keys())[:-3]  # Exclude "accuracy", "macro avg", "weighted avg"
precision = [report[cat]["precision"] for cat in categories]
recall = [report[cat]["recall"] for cat in categories]
f1_score = [report[cat]["f1-score"] for cat in categories]

# Step 5: Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the data points and decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.title(f"SVM Decision Boundary (Accuracy: {accuracy:.2f})")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.colorbar(label="Class")
plt.show()

# Step 6: Plot metrics
x = np.arange(len(categories))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label="Precision", color='blue')
plt.bar(x, recall, width, label="Recall", color='orange')
plt.bar(x + width, f1_score, width, label="F1-Score", color='green')

plt.xticks(x, categories)
plt.title("Classification Metrics")
plt.xlabel("Class")
plt.ylabel("Score")
plt.legend()
plt.show()

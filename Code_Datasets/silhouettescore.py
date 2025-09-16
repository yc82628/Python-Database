import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

# Data points and cluster labels
data_points = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
cluster_labels = [0, 0, 0, 1, 1, 1]

# Calculate silhouette score
score = silhouette_score(data_points, cluster_labels)
print(f"Silhouette Score: {score:.2f}")

# Convert data to numpy array for plotting
data_points = np.array(data_points)

# Plot clusters
plt.figure(figsize=(8, 6))
for label in np.unique(cluster_labels):
    cluster_points = data_points[np.array(cluster_labels) == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', s=100)

# Mark cluster centers
cluster_centers = [[1, 2], [10, 2]]  # Replace with calculated centers if needed
cluster_centers = np.array(cluster_centers)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', label='Cluster Centers', s=150, marker='X')

plt.title("Cluster Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid()
plt.show()

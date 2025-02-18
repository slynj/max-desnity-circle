import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from scipy.spatial import distance

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)

db = DBSCAN(eps=0.3, min_samples=5).fit(X)
labels = db.labels_

unique_labels = set(labels) - {-1} 
centers = []

for l in unique_labels:
    mask_bool = (labels == l)
    cluster_points = X[mask_bool]

    x_min = np.min(cluster_points[:, 0])
    x_max = np.max(cluster_points[:, 0])
    y_min = np.min(cluster_points[:, 1])
    y_max = np.max(cluster_points[:, 1])

    center_pt = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    centers.append((l, center_pt))

radius = 1
pt_counts = {}

for l, c in centers:
    count = np.sum(distance.cdist([c], X, 'euclidean')[0] <= radius)
    pt_counts[l] = count
    print(f"cluster {l}: {c} [{count}]")

max_dens_cluster = max(pt_counts, key=pt_counts.get)
max_dens_pt = dict(centers)[max_dens_cluster]

plt.figure(figsize=(8, 6))     
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')

for l, c in centers:
    markerC = 'red' if l != max_dens_cluster else 'blue'
    plt.scatter(c[0], c[1], c=markerC, marker='X', s=200, label=f"Cluster {l}")

    circle = plt.Circle(c, radius, color=markerC, fill=False, linestyle='dashed')
    plt.gca().add_patch(circle)

plt.axis('equal')
plt.title("DBSCAN Clustering - Max Density Circle")
plt.legend()
plt.show()
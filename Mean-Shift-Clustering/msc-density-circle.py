import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from scipy.spatial import distance

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)
labels = ms.labels_
centers = ms.cluster_centers_

radius = 1
pt_counts = {}

for l, c in enumerate(centers):
    count = np.sum(distance.cdist([c], X, 'euclidean')[0] <= radius)
    pt_counts[l] = count
    print(f"cluster {l}: {c} [{count}]")

max_dens_cluster = max(pt_counts, key=pt_counts.get)
max_dens_pt = centers[max_dens_cluster]

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

for l, c in enumerate(centers):
    markerC = 'red' if l != max_dens_cluster else 'blue'
    plt.scatter(c[0], c[1], c=markerC, marker='X', s=200, label=f"cluster {l}")

    circle = plt.Circle(c, radius, color=markerC, fill=False, linestyle='dashed')
    plt.gca().add_patch(circle)

plt.axis('equal')
plt.title("Mean Shift Clustering - Max Density Circle")
plt.legend()
plt.show()
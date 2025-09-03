import torch
import matplotlib.pyplot as plt

from torchclust.utils.datasets import make_blobs
from torchclust.centroid import KMeans
from torchclust.density import DBSCAN

x, _ = make_blobs(1000, num_features=2, num_centers=10)

kmeans = KMeans(num_clusters=10)
labels = kmeans.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=labels)
plt.show()
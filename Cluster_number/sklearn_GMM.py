import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture  # 高斯混合模型
import pandas as pd
import numpy as np

centers = [[1, 1], [-1, -1], [1, -1],[-1, 1]]
Xn, labels_true = make_blobs(n_samples=80, centers=centers, cluster_std=0.5, random_state=2)

model = GaussianMixture(n_components=3)
y_pred = model.fit_predict(Xn)
print(y_pred)

plt.figure('GMM', facecolor='lightgray')
plt.title('GMM', fontsize=16)
plt.tick_params(labelsize=10)
plt.scatter(Xn[:, 0], Xn[:, 1], s=80, c=y_pred, cmap='brg', label='Samples')
plt.legend()
plt.show()
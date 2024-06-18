from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=50, n_features=2, centers=6, random_state=2)
km_model = KMeans(n_clusters=6, max_iter=20)
km_model.fit(x)
y_predict = km_model.predict(x)
# 绘制聚类好的数据
plt.scatter(x[:, 0], x[:, 1], c=y_predict, alpha=0.5, s=50)
# 获取并绘制簇心
centers = km_model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100)
plt.show()
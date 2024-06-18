import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def multiGaussian(x, mu, sigma):
    return 1 / ((2 * np.pi) * pow(np.linalg.det(sigma), 0.5)) * np.exp(
        -0.5 * (x - mu).dot(np.linalg.pinv(sigma)).dot((x - mu).T))

def computeGamma(x, mu, sigma, alpha):
    n_samples = x.shape[0]
    n_clusters = len(alpha)
    gamma = np.zeros((n_samples, n_clusters))
    p = np.zeros(n_clusters)
    g = np.zeros(n_clusters)
    for i in range(n_samples):
        for j in range(n_clusters):
            p[j] = multiGaussian(x[i], mu[j], sigma[j])
            g[j] = alpha[j] * p[j]
        for k in range(n_clusters):
            gamma[i, k] = g[k] / np.sum(g)
    return gamma


class GMM1:
    def __init__(self, n_clusters, ITER=100):
        self.n_clusters = n_clusters
        self.ITER = ITER
        self.mu = []
        self.sigma = []
        self.alpha = []

    def fit(self, data):
        n_samples = data.shape[0]
        n_features = data.shape[1]
        alpha = np.ones(self.n_clusters) / self.n_clusters
        mu = data[np.random.choice(range(n_samples), self.n_clusters)]
        sigma = np.full((self.n_clusters, n_features, n_features), np.diag(np.full(n_features, 0.1)))
        for k in range(self.ITER):
            gamma = computeGamma(data, mu, sigma, alpha)
            alpha = np.sum(gamma, axis=0) / n_samples
            for i in range(self.n_clusters):
                mu[i] = np.sum(data * gamma[:, i].reshape((n_samples, 1)), axis=0) / np.sum(gamma, axis=0)[i]
                sigma[i] = 0
                for j in range(n_samples):
                    sigma[i] += (data[j].reshape((1, n_features)) - mu[i]).T.dot(
                        (data[j] - mu[i]).reshape((1, n_features))) * gamma[j, i]
                sigma[i] = sigma[i] / np.sum(gamma, axis=0)[i]
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha

    def predict(self, data):
        pred = computeGamma(data, self.mu, self.sigma, self.alpha)
        cluster_results = np.argmax(pred, axis=1)
        return cluster_results

if __name__ == "__main__":
    X, Y = make_blobs(n_samples=200, n_features=2, centers=5, cluster_std=1.0, random_state=1)
    model1 = GMM1(5, 100)
    model1.fit(X)
    result = model1.predict(X)
    # print("\nmu:", model1.mu, "\nsigma:", model1.sigma, "\nalpha:", model1.alpha)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    GMM = GaussianMixture(n_components=5, covariance_type='full', init_params='random').fit(X)
    # print('weights_:', GMM.weights_, '\nmeans_:', GMM.means_, '\ncovariances_:', GMM.covariances_)
    plt.show()
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
import copy
import numpy as np
import pandas as pd
import random

# 计算距离
def distance(x, y):
    dis = 0
    for i in range(0, len(x) - 1):
        dis += pow((x[i] - y[i]), 2)
    return np.sqrt(dis)

# 检查是否需要继续迭代
def iteration(last_cluster, current_cluster):
    # 如果长度不一样
    if len(last_cluster) != len(current_cluster):
        return False
    lenc = len(current_cluster)
    for i in range(0, lenc):
        # 如果元素个数不一样
        if len(last_cluster[i]) != len(current_cluster[i]):
            return False
        lenl = len(last_cluster[i])
        # 如果元素不一样
        for j in range(0, lenl):
            if last_cluster[i][j] != current_cluster[i][j]:
                return False
    return True

# k均值算法
def kmeans(data, K):
    lent = len(data)
    lens = lent // K
    # 随机初始化 K 个聚类中心
    for i in range(0, K):
        centers.append(data[random.randint(i * lens, (i + 1) * lens - 1)])
    clusters = []
    for i in range(0, K):
        clusters.append([])
    last_clusters = []
    # 只要还能迭代就继续迭代，用 K-means 方法计算新的聚类中心
    while not iteration(last_clusters, clusters):
        last_clusters = copy.deepcopy(clusters)
        for i in range(0, K):
            clusters[i].clear()
        dis = []
        for i in range(0, K):
            dis.append([])

        for i in range(0, lent):
            for j in range(0, K):
                dis[j].append(distance(centers[j], data[i]))

        for i in range(0, lent):
            max_dis = float("inf")
            max_dis_id = -1
            for j in range(0, K):
                if dis[j][i] < max_dis:
                    max_dis = dis[j][i]
                    max_dis_id = j
            clusters[max_dis_id].append(data[i])

        new_center_1 = []
        new_center_2 = []
        new_center_3 = []
        new_center_4 = []

        len_clusters = []
        for i in range(0, K):
            len_clusters.append(len(clusters[i]))
            new_center_1.append(0)
            new_center_2.append(0)
            new_center_3.append(0)
            new_center_4.append(0)

        for i in range(0, K):
            for j in range(0, len_clusters[i]):
                new_center_1[i] += clusters[i][j][0]
                new_center_2[i] += clusters[i][j][1]
                new_center_3[i] += clusters[i][j][2]
                new_center_4[i] += clusters[i][j][3]

        for i in range(0, K):
            new_center_1[i] /= len_clusters[i]
            new_center_2[i] /= len_clusters[i]
            new_center_3[i] /= len_clusters[i]
            new_center_4[i] /= len_clusters[i]
            centers[i] = [new_center_1[i], new_center_2[i],new_center_3[i],new_center_4[i],i]
    return last_clusters

if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:150, [0, 1, 2, 3, 4]])

    floder = KFold(n_splits=5, shuffle=True, random_state=7)  # 初始化KFold
    train_files = []  # 存放5折的训练集划分
    test_files = []  # # 存放5折的测试集集划分

    # 5折交叉验证
    for k, (Trindex, Tsindex) in enumerate(floder.split(data)):
        train_files.append(np.array(data)[Trindex].tolist())
        test_files.append(np.array(data)[Tsindex].tolist())

    matrix = np.zeros((3, 3))

    for i in range(5):
        centers = []

        clusters = kmeans(train_files[i], 3)

        for j in range(30):
            max_dis = float("inf")
            max_dis_id = -1
            for k in range(3):

                dis = distance(centers[k],test_files[i][j])
                if dis < max_dis:
                    max_dis = dis
                    max_dis_id = centers[k][4]

            # 构造矩阵，计算ACC等
            if test_files[i][j][4] == 0:
                if max_dis_id == 0:
                    matrix[0][0] += 1
                elif max_dis_id == 1:
                    matrix[0][1] += 1
                elif max_dis_id == 2:
                    matrix[0][2] += 1
            elif test_files[i][j][4] == 1:
                if max_dis_id == 0:
                    matrix[1][0] += 1
                elif max_dis_id == 1:
                    matrix[1][1] += 1
                elif max_dis_id == 2:
                    matrix[1][2] += 1
            elif test_files[i][j][4] == 2:
                if max_dis_id == 0:
                    matrix[2][0] += 1
                elif max_dis_id == 1:
                    matrix[2][1] += 1
                elif max_dis_id == 2:
                    matrix[2][2] += 1

    sumrow=[0,0,0]
    sumcol=[0,0,0]
    sumrow[0] = matrix[0][0] + matrix[0][1] + matrix[0][2]
    sumrow[1] = matrix[1][0] + matrix[1][1] + matrix[1][2]
    sumrow[2] = matrix[2][0] + matrix[2][1] + matrix[2][2]
    sumcol[0] = matrix[0][0] + matrix[1][0] + matrix[2][0]
    sumcol[1] = matrix[0][1] + matrix[1][1] + matrix[2][1]
    sumcol[2] = matrix[0][2] + matrix[1][2] + matrix[2][2]
    sum = sumcol[0] + sumcol[1] + sumcol[2]

    # 计算ACC等
    Acc = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / sum
    print("ACC为:", Acc)
    pre = [0, 0, 0]
    recall = [0, 0, 0]
    for i in range (3):
        pre[i] = matrix[i][i] / sumcol[i]
        recall[i] = matrix[i][i] / sumrow[i]
        print("第", i + 1, "类的Pre为：",pre[i])
        print("第", i + 1, "类的recall为：",recall[i])

    print("ap为:", (recall[0] + recall[1] + recall[2]) / 3)
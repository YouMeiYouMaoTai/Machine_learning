import numpy as np
import matplotlib.pyplot as plt

# 计算欧式距离
def dist_oushi(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 手写的二维数据集
def getData_txt():
    dataSet = []
    fileIn = open('D:\\Desktop\\Program\\Python\\Cluster_number\\DataSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
    return dataSet

# 初始化簇中心点 一开始随机从样本中选择k个 当做各类簇的中心
def Init_Centre(data, k):
    num, dim = data.shape
    centpoint = np.zeros((k, dim))
    l = [x for x in range(num)]
    np.random.shuffle(l)
    for i in range(k):
        index = int(l[i])
        centpoint[i] = data[index]
    return centpoint

# 进行KMeans分类
def KMeans(data, k):
    # 样本个数
    num = np.shape(data)[0]
    # 记录各样本 簇信息 0:属于哪个簇 1:距离该簇中心点距离
    cluster = np.zeros((num, 2))
    cluster[:, 0] = -1

    # 记录是否有样本改变簇分类
    change = True
    # 初始化各簇中心点
    cp = Init_Centre(data, k)

    while change:
        change = False

        # 遍历每一个样本
        for i in range(num):
            minDist = 9999.9
            minIndex = -1

            # 计算该样本距离每一个簇中心点的距离 找到距离最近的中心点
            for j in range(k):
                dis = dist_oushi(cp[j], data[i])
                if dis < minDist:
                    minDist = dis
                    minIndex = j

            # 如果找到的簇中心点非当前簇 则改变该样本的簇分类
            if cluster[i, 0] != minIndex:
                change = True
                cluster[i, :] = minIndex, minDist

        # 根据样本重新分类  计算新的簇中心点
        for j in range(k):
            pointincluster = data[[x for x in range(num) if cluster[x, 0] == j]]
            cp[j] = np.mean(pointincluster, axis=0)

    print("已完成，结果如图所示：")
    return cp, cluster

# 展示结果  各类簇使用不同的颜色  中心点使用X表示
def Show(data, k, cp, cluster):
    num, dim = data.shape
    # 分别用不同的颜色来显示点
    color = ['r', 'g', 'b', 'y','c','k','m']
    # 二维图
    for i in range(num):
        mark = int(cluster[i, 0])
        plt.plot(data[i, 0], data[i, 1], color[mark] + 'o')
    for i in range(k):
        plt.plot(cp[i, 0], cp[i, 1], color[i] + 'x')
    plt.show()

print("输入需要几个簇中心")
num = int(input())
data = np.array(getData_txt())
cp, cluster = KMeans(data, num)
Show(data, num, cp, cluster)

'''
# 随机产生n个dim维度的数据
def DataSet(n, dim):
    data = []
    while len(data) < n:
        p = np.around(np.random.rand(dim) * 30, decimals=2)
        data.append(p)
    return data
'''
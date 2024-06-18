from numpy import inf
import pylab as pl
import math

def dist(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))

def dist_av(Ci, Cj):
    return sum(dist(i, j) for i in Ci for j in Cj)/(len(Ci)*len(Cj))

# 获取数据集
def getData_txt():
    dataSet = []
    fileIn = open('DataSet\\SVM_test.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append((float(lineArr[0]), float(lineArr[1])))
    return dataSet

#找到距离最小的下标
def find_Min(Dist):
    min = inf
    x = 0; y = 0
    for i in range(len(Dist)):
        for j in range(len(Dist[i])):
            if i != j and Dist[i][j] < min:
                min = Dist[i][j];x = i; y = j
    return (x, y, min)

#算法模型：
def Layer(dataset, dist, k):
    #初始化 坐标矩阵 和 距离矩阵
    Coordinate = []
    Dist = []
    for i in dataset:
        Ci = []
        Ci.append(i)
        Coordinate.append(Ci)
    for i in Coordinate:
        Mi = []
        for j in Coordinate:
            Mi.append(dist(i, j))
        Dist.append(Mi)
    q = len(dataset)
    #合并更新
    while q > k:
        x, y, min = find_Min(Dist)
        Coordinate[x].extend(Coordinate[y])
        Coordinate.remove(Coordinate[y])
        Dist = []
        for i in Coordinate:
            Mi = []
            for j in Coordinate:
                Mi.append(dist(i, j))
            Dist.append(Mi)
        q -= 1
    return Coordinate

#画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []    #x坐标列表
        coo_Y = []    #y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color = colValue[i%len(colValue)], label=i)
    pl.legend(loc='upper right')
    pl.show()

if __name__ == "__main__":
    print("数据集:\n", getData_txt())
    Coordinate = Layer(getData_txt(), dist_av, 5)
    draw(Coordinate)

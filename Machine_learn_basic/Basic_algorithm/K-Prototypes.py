import numpy as np

# 计算欧式距离
def dist_oushi(x, y):
    return np.sqrt(sum(pow((x[i] - y[i]), 2) for i in range(4)))

# 计算汉明距离
def dist_Ham(x, y):
    return int(x[4] != y[4])

# 返回总距离，字符距离权值设置为 2
def distance(x, y):
    return dist_oushi(x, y) + 2 * dist_Ham(x, y)

# 手写的二维数据集
def getData():
    dataSet = [[2, 5, 3, 6, 'a'], [3, 1, 4, 6, 'a'],
               [5, 2, 3, 1, 'a'], [8, 6, 5, 2, 'a'],
               [9, 6, 2, 7, 'b'], [7, 3, 2, 6, 'b'],
               [0, 4, 7, 2, 'b'], [8, 3, 4, 1, 'b'],
               [6, 6, 7, 2, 'c'], [3, 2, 7, 8, 'c'],
               [8, 4, 5, 2, 'c'], [3, 5, 6, 1, 'c'],
               [7, 3, 6, 0, 'd'], [5, 2, 6, 7, 'd'],
               [5, 2, 6, 8, 'd'], [7, 1, 2, 6, 'd']]
    return dataSet

# 初始化簇中心点 一开始随机从样本中选择 k 个 当做各类簇的中心
def Init_Centre(data, k):
    centpoint = []
    l = [x for x in range(len(data))]
    np.random.shuffle(l)
    for i in range(k):
        centpoint.append(data[l[i]])
    return centpoint

'''
data  = getData()
print(data[0],'\n\n', data[4])
print(distance(data[0], data[4], 2))
'''

# 返回新的聚类中心
def cluster_new(data, cluster, no):
    num_av = [0, 0, 0, 0]
    char_most_list = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    char_most_num = -1
    char_most = ''
    coincident = []     # 同一类的 data[i] 的 i
    centrepoint = []    # 目前新一类的所有平均值
    dis = {}    # 同一类与新一类的距离，key表示data[i]中的i，value表示距离
    for i in range(len(cluster)):
        if cluster[i][0] == no:
            coincident.append(i)
            dis[i] = 0

            # 统计字符的众数
            for char in char_most_list:
                if char == data[i][4]:
                    char_most_list[char] += 1

            # 统计数值型的平均数
            for j in range(4):
                num_av[j] += (data[i][j])

    # 统计数值的平均值，字符的众数
    for i in range(len(num_av)):
        num_av[i] /= len(coincident)
    for char in char_most_list:
        if char_most_list[char] > char_most_num:
            char_most = char
            char_most_num = char_most_list[char]
    centrepoint = num_av
    centrepoint.append(char_most)
    # 统计这一类和平均的距离
    for i in range(len(coincident)):
        dis[coincident[i]] = distance(data[coincident[i]], centrepoint)
    min_dis = 999
    min_num = 0
    for num in dis:
        if dis[num] < min_dis:
            min_dis = dis[num]
            min_num = num
    return data[min_num]

# 进行 K_Prototypes 分类
def K_Prototypes(data, k):
    # 样本个数
    num = len(data)
    # 记录各样本簇信息 0:属于哪个簇 1:距离该簇中心点距离
    cluster = np.zeros((num, 2))
    cluster[:, 0] = -1

    # 记录是否有样本改变簇分类
    change = True
    # 初始化各簇中心点
    cp = Init_Centre(data, k)
    print("初始选择的中心为：\n", cp)

    while change:
        change = False

        # 遍历每一个样本
        for i in range(num):
            minDist = 9999.9
            minIndex = -1

            # 计算该样本距离每一个簇中心点的距离 找到距离最近的中心点
            for j in range(k):
                # 设置的符号类型权重为 2
                dis = distance(cp[j], data[i])
                if dis < minDist:
                    minDist = dis
                    minIndex = j

            # 如果找到的簇中心点非当前簇 则改变该样本的簇分类
            if cluster[i, 0] != minIndex:
                change = True
                cluster[i, :] = minIndex, minDist

        # 根据样本重新分类  计算新的簇中心点
        for j in range(k):
            cp[j] = cluster_new(data, cluster, j)

    return cp, cluster

if __name__ == "__main__":
    print("请输入想要几个聚类中心：")
    k = int(input())
    cp, cluster = K_Prototypes(getData(), k)
    print("数据集：\n", getData())
    print("聚类后选择的中心的为：\n", cp)
    print("各数据点聚类结果为：（前为类别，后为和中心的距离）\n", cluster)
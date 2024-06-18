from math import log

# 计算信息熵
def entropy_calculate(dataSet):
    num = len(dataSet)  # 数据集的样本数
    label_category = {}   # 西瓜属性的类别数组
    entropy0 = 0
    # 统计当前属性每个类别的个数
    for feature in dataSet:  # 遍历每个样本
        Label_now = feature[-1]  # 获取当前样本的属性
        if Label_now not in label_category.keys():
            label_category[Label_now] = 0   # 如果属性类别不存在，初始化该属性类别
        label_category[Label_now] += 1
    # 计算该属性的信息熵
    for p in label_category:
        Pk = label_category[p] / num
        entropy0 = entropy0 - Pk * log(Pk, 2)
    # 返回信息熵
    return entropy0

# 划分数据集
def dataset_split(dataSet, axis, value):   # axis是属性索引，value是返回的子集对应的属性值
    retDataSet = []   # 划分后的数据集
    for feature in dataSet:
        if feature[axis] == value:   # 如果该属性存在
            # 以该属性为划分点
            reducedFeatVec = feature[:axis]
            reducedFeatVec.extend(feature[axis + 1:])
            retDataSet.append(reducedFeatVec)
    # 返回划分后的数据集
    return retDataSet

# 选取最好的数据集划分方式
def dataset_split_best(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = entropy_calculate(dataSet)
    bestInfoGainRatio = 0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        label_category = set(featList)  # get a set of unique values
        newEntropy = 0.0
        splitInfo = 0.0
        for value in label_category:
            subDataSet = dataset_split(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * entropy_calculate(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if infoGain == 0:   # fix the overflow bug
            continue
        infoGainRatio = infoGain / splitInfo
        if infoGainRatio > bestInfoGainRatio:  # compare this to the best gain so far
            bestInfoGainRatio = infoGainRatio  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer

# 找出出现次数最多的类别
def most_category(classList):
    label_category = {}
    # 统计该属性中每个类别出现的次数
    for Label_now in classList:
        if Label_now not in label_category.keys():
            label_category[Label_now] = 0   # 初始化属性类别
        label_category[Label_now] += 1
    # 通过排序找出次数最多者
    Category_Sorted = sorted(label_category.items())
    # 返回出现次数最多的类别
    return Category_Sorted[0][0]

# 递归法生成决策树
def tree_create(dataSet, labels, featLabels_2):
    classList = [example[-1] for example in dataSet]  # 类别向量
    if classList.count(classList[0]) == len(classList):  # 如果只有一个类别，返回
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类别
        return most_category(classList)
    Best_Feature = dataset_split_best(dataSet)  # 最优划分属性的索引
    bestFeatLabel = labels[Best_Feature]  # 最优划分属性的标签
    featLabels_2.append(bestFeatLabel)
    tree = {bestFeatLabel: {}}
    del (labels[Best_Feature])  # 已经选择的特征不再参与分类
    featValues = [example[Best_Feature] for example in dataSet]
    uniqueValue = set(featValues)  # 该属性所有可能取值，也就是节点的分支
    for value in uniqueValue:  # 对每个分支，递归构建树
        subLabels = labels[:]
        tree[bestFeatLabel][value] = tree_create(dataset_split(dataSet, Best_Feature, value), subLabels, featLabels_2)
    return tree, featLabels_2

# 测试
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))             #获取决策树结点
    secondDict = inputTree[firstStr]             #下一个字典
    featIndex = featLabels.index(firstStr)
    classLabel = ""
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            classLabel = classify(secondDict[key], featLabels, testVec)
        else:
            classLabel = secondDict[key]
    return classLabel
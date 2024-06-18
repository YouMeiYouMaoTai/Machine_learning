from math import log
# 返回香农熵
def initial_Ent(DataSet):
    row_DataSet = len(DataSet)                        #返回数据集的行数
    num_label = {}                                #保存每个标签(Label)出现次数的字典
    shannonEnt = 0.0                                #经验熵(香农熵)
    for featVec in DataSet:                            #对每组特征向量进行统计
        current_Label = featVec[-1]                    #提取标签(Label)信息
        if current_Label not in num_label.keys():    #如果标签(Label)没有放入统计次数的字典,添加进去
            num_label[current_Label] = 0
        num_label[current_Label] += 1                #Label计数
    for key in num_label:                            #计算香农熵
        rate = float(num_label[key]) / row_DataSet    #选择该标签(Label)的概率
        shannonEnt -= rate * log(rate, 2)            #利用公式计算
    return shannonEnt                               #返回经验熵(香农熵)

# 按照给定特征划分数据集
def split_DataSet(dataSet, number, value):
    retDataSet = []                                        #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        if featVec[number] == value:
            reducedFeatVec = featVec[:number]                #去掉number特征
            reducedFeatVec.extend(featVec[number+1:])     #将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最优特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                    #特征数量
    baseEntropy = initial_Ent(dataSet)                 #计算数据集的香农熵
    bestInfoGain = 0.0                                  #信息增益
    bestFeature = -1                                    #最优特征的索引值
    for i in range(numFeatures):                         #遍历所有特征
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                    #创建set集合{},元素不可重复,表示有多少种分支
        newEntropy = 0.0                              #经验条件熵
        for value in uniqueVals:                      #计算信息增益
            subDataSet = split_DataSet(dataSet, i, value)         #subDataSet划分后的子集
            rate = len(subDataSet) / float(len(dataSet))           #计算子集的概率
            newEntropy += rate * initial_Ent(subDataSet)     #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                     #信息增益
        if infoGain > bestInfoGain:                           #通过比较找到最大的那一个
            bestInfoGain = infoGain                             #更新信息增益，找到最大的信息增益
            bestFeature = i                                     #记录信息增益最大的特征的索引值
    return bestFeature                                             #返回信息增益最大的特征的索引值

# 统计此处出现最多的元素
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items())        #根据字典的值降序排序
    return sortedClassCount[0][0]                                #返回classList中出现次数最多的元素

# 创建决策树
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]            #取分类标签(好瓜还是坏瓜)
    if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                    #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #选择最优特征
    bestFeatLabel = labels[bestFeat]                            #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}                                    #根据最优特征的标签生成树
    del(labels[bestFeat])                                        #删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]        #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                    #遍历特征，创建决策树。
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(split_DataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree, featLabels

# 测试 ID3 算法的结果
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
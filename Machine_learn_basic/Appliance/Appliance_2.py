import os
import re
import jieba
import jieba.analyse
from collections import Counter

# 获取文章的名称
def getfile_name():
    file_name = []
    path = os.getcwd() + '\\DataSet\\data'
    for i in os.listdir(path):
        file_name.append(i)
    return file_name

# 获取txt文本，变成一个数组返回
def getData_txt():
    dataSet = []
    path = os.getcwd() + '\\DataSet\\data'
    for i in os.listdir(path):
        fileIn = open(path + '\\' + i, 'r', encoding = 'utf-8')
        data = fileIn.read()
        dataSet.append(data)
    return dataSet

# 把每篇文章分词并放入数组中
def jieba_data(DataSet):
    DataSet_jieba = []
    # 只留中文
    p = re.compile("[^\u4e00-\u9fa5^]")
    for txt in DataSet:
        txt = re.sub(p, '', txt)
        # jieba.load_userdict("Jieba_dict.txt")
        seg_list = jieba.lcut(txt)
        DataSet_jieba.append(seg_list)
    return DataSet_jieba

# 统计每篇文章里面的关键字的个数
def stastic(dataset, str_list):
    word_count_total = []
    key = []
    for data in dataset:
        word_num = {}
        # 初始化字典
        for i in str_list:
            word_num[i] = 0
            key.append(i)
        for i in range(len(data)):
            for j in range(len(key)):
                if key[j] == data[i]:
                    word_num[key[j]] += 1
        word_count_total.append(word_num)
    return word_count_total

# 寻找最相关的文章的序号, 参数是一个列表，列表的元素是字典
def research(word_count_total):
    # key记录是第几篇文章，value记录这篇文章有几个关键字对应
    word_num = {}
    # 初始化字典
    for i in range(len(word_count_total)):
        word_num[i] = 0
    # 记录第 no 篇文章的 value 总合
    no = 0
    # 每篇文章的 value 总和
    for dict in word_count_total:
        value_sum = 0
        for key in dict:
            value_sum += dict[key]
        word_num[no] = value_sum
        no += 1
    # 对字典进行 value 值的从大到小排序
    sort_dict = sorted(word_num.items(), key=lambda x: x[1], reverse=True)
    # value总和最大的前三篇文章的下标
    max_no = []
    epoch = 0   # 只要 3 篇
    for i in sort_dict:
        max_no.append(i[0])
        epoch += 1
        if epoch == 3:
            break
    return max_no

# 把杂乱的分词结果中的非关键信息剔除
def remove_trash(seg_list):
    # 打开写入关键词的文件
    keyword = open(os.getcwd() + '\\keyword.txt', 'w+', encoding='utf-8')
    wordlist = []

    # 获取停用词表
    stop = open(os.getcwd() + '\\Jieba_dict.txt', 'r+', encoding='utf-8')
    # 用‘\n’去分隔读取，返回一个一维数组
    stopword = stop.read().split("\n")
    # 遍历分词表
    for key in seg_list.split('/'):
        # 去除停用词，去除单字
        if not(key.strip() in stopword) and (len(key.strip()) > 1):
            wordlist.append(key)
            keyword.write(key + "\n")

    stop.close()
    keyword.close()
    return wordlist

# 获取测试文章的关键词列表
def get_essence(proto_paper):
    fp = open(proto_paper, encoding='utf-8')
    data_test = fp.read()

    # 只要中文
    p = re.compile("[^\u4e00-\u9fa5^]")
    data_test = re.sub(p, '', data_test)
    seg_list = jieba.lcut(data_test, cut_all=True)  # 分词

    # 返回一个以分隔符'/'连接各个元素后生成的字符串
    line = "/".join(seg_list)
    word = remove_trash(line)

    return word

# 获取前十个重复最多的关键词列表
def keyword_top10(proto_paper):
    paper_word = get_essence(proto_paper)
    # 取前十个重复次数最多的关键字
    keyword_top10_item = Counter(paper_word).most_common(10)
    keyword_top10 = []
    for i in keyword_top10_item:
        keyword_top10.append(i[0])
    return keyword_top10

if __name__ == '__main__':
    # 文献名列表
    file_name = getfile_name()
    # 分词前的整篇文章
    dataset_pre = getData_txt()
    # 分词后的关键词列表
    dataset_after = jieba_data(dataset_pre)

    str_list = ["机器学习", "神经网络", "特征提取"]
    # 存放 30 篇文章关键字字典的列表
    word_count_total_func2 = stastic(dataset_after, str_list)

    # 最相关的三篇文章的下标数组
    max_paper_func2 = research(word_count_total_func2)
    print("与关键词",str_list, "最相关的前三篇文章为：")
    for i in max_paper_func2:
        print(file_name[i][0:-4])

    print("-----------------------------------------------------------")

    # 给定一篇文章，返回最相似的前三篇
    proto_paper = os.getcwd() + '\\DataSet\\data\\基于机器学习训练金属离子吸附能预测模型的研究.txt'
    # 获取分词列表
    keyword_top10 = keyword_top10(proto_paper)
    word_count_total_func3 = stastic(dataset_after, keyword_top10)

    # 最相似的三篇文章的下标数组
    max_paper_func3 = research(word_count_total_func3)
    print("与", proto_paper[63:-4], "最相似的前三篇文章为：")
    for i in max_paper_func3:
        print(file_name[i][0:-4])
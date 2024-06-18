# 主函数模块

import os
import Function

'''
1、遍历处理训练集，把训练集图片得到的字符串类型数据放到一个 txt 文件里面存储
    1）、存储的时候要同时把数据归属的数字类型存储下来，即 每个训练集文件名称 ".bmp" 前的字符
    2）、存储完每一个数字数据的时候要 ‘\n’
2、遍历处理测试集，把每张图片得到的数据和整个训练集得到的 txt 文件数据进行 欧氏距离 计算
找到最小的那一个进行统计，如果有多个最小的，则对比多个匹配的数字是否为同一个数字
3、每识别一个就输出一个，最终统计正确率
'''

# 训练集、测试集、数据存储文件 的地址
Datafile_train = "D:\\desktop\\Program\\Python\\Number_Match\\Data_Set\\train-images"
Datafile_test = "D:\\desktop\\Program\\Python\\Number_Match\\Data_Set\\test-images"
Datafile_number = "D:\\desktop\\Program\\Python\\Number_Match\\numberData.txt"

# 写模式打开 txt 文件
numberData_file = open(Datafile_number, 'w')

# 遍历处理训练集，得到模板库
for name_train in os.listdir(Datafile_train):
    # path_train 为训练集图片完整路径
    path_train = Datafile_train + "\\" + name_train
    # 出去数据以外，把数据归属的数字也保存下来
    img_str = name_train[0] + ":" + Function.img_handle(path_train)
    # 保存完一个数字数据之后换行
    numberData_file.write(img_str + "\n")
numberData_file.close()

# 读模式打开 txt 文件
numberData_file = open(Datafile_number, 'r')

# 统计测试集图片和被正确识别图片的数量，用来计算正确率、错误率和拒绝识别率
img_count = [0 for i in range(10)]
sure_count = [0 for i in range(10)]
refuse_count = [0 for i in range(10)]

# 遍历处理测试集，每张图片与模板库进行对比
for name_test in os.listdir(Datafile_test):  # 读一张测试集图片
    # 每识别一个数字就把相对应的识别数量 +1
    img_count[eval(name_test[0:1])] += 1
    # path_test 为测试集图片完整路径
    path_test = Datafile_test + "\\" + name_test
    min_dist = 255      # 假定的最小欧氏距离
    test_str = Function.img_handle(path_test)   # 测试图片的数据
    img_number = ""     # 标明测试结果的数字归属
    # 每次读取模板库的一行数据
    number_str = numberData_file.readline()
    while number_str:
        train_str = number_str[-50:-1]  # 只要后49个数据
        dist = Function.distance(test_str, train_str)
        if min_dist > dist:
            min_dist = dist
            img_number = number_str[0:1]  # 标记为对应的数字
        elif min_dist == dist:
            if name_test[0:1] != img_number[len(img_number) - 1]:
                img_number += name_test[0:1]    # 把相等距离的数都放进字符串里
        number_str = numberData_file.readline()
    numberData_file.seek(0)
    if len(img_number) == 1:    # 如果img_number里只有一个数字
        if img_number == name_test[0:1]:    # 识别正确
            sure_count[eval(name_test[0:1])] += 1
            print("测试数字：", name_test[0:1], " 识别结果为：", img_number[0], end="  ")
            print("---True")
        else:    # 识别错误
            print("测试数字：", name_test[0:1], " 识别结果为：", img_number[0], end="  ")
            print("---False")
    else:   # 如果img_number里有多个数字
        NUMBER = 0     # 用来标记遍历到 img_number 的第几个了
        similar = 1     # 用来标记 img_number 里面的数字是否都是同一个
        for i in img_number:
            if NUMBER < len(img_number) - 1:
                NUMBER += 1
            if i != img_number[NUMBER]:
                similar = 0
                break
        if similar:     # 如果 img_number 里面的数字都是同一个
            sure_count[eval(name_test[0:1])] += 1
            print("测试数字：", name_test[0:1], " 识别结果为：", img_number[0], end="  ")
            print("---正确")
        else:       # 如果 img_number 里面的数字不是同一个
            refuse_count[eval(name_test[0:1])] += 1
            print("测试数字：", name_test[0:1], " 识别结果为：", end=" ")
            for i in range(len(img_number)):
                print(img_number[i], end=" ")
            print("---有多个类别的模板匹配，无法识别！")
numberData_file.close()

# 计算百分率
for i in range(len(img_count)):
    sure_count[i] *= 100
    refuse_count[i] *= 100

print("————————————————————————————————————————————————————————————")
for i in range(10):
    print("数字 {:d} 识别的正确率 = {:.2f}% ，错误率 = {:.2f}% ，拒绝识别率 = {:.2f}%"
          .format(i, sure_count[i] / img_count[i], 100 - sure_count[i] / img_count[i] - refuse_count[i] / img_count[i], refuse_count[i] / img_count[i]))


# 功能函数模块

import cv2
import numpy as np

'''
原图片有 28 * 28 个像素点，分成 7 * 7 个小图片，每个小图片有 4 * 4 个像素点
先当成 7 * 7 的二维矩阵，再处理二维矩阵中的每一个元素里包含的像素
'''
# 处理图像获得 字符串 类型数据
def img_handle(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    data_str = ""    # 字符串 类型保存图像数据
    for a in range(7):
        for b in range(7):
            count = 0   # 统计满足要求的像素点的数量
            for c in range(4):
                for d in range(4):
                    if img[a * 4 + c][b * 4 + d] >= 127:
                        count += 1
            # 经过多次测试，count >= 5 的时候识别正确率最高
            if count >= 5:
                data_str += "1"
            else:
                data_str += "0"
    return data_str

# 求两个字符串之间的欧氏距离
def distance(img_str1, img_str2):
    dist = np.sqrt(sum([np.square(eval(img_str1[i:i + 1]) - eval(img_str2[i:i + 1]))
                        for i in range(len(img_str1))]))
    return dist

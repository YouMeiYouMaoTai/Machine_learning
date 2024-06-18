import cv2
import os
import numpy as np

Datafile_train = "D:\\desktop\\Program\\Python\\Face_Recognition\\ORL_Faces"
List = []       # 存放图像数据, 元素为 列表[], 每个元素大小为 112 x 92 = 10304

# 把每一张图片的地址读取出来
for name_file in os.listdir(Datafile_train):
    path_file = Datafile_train + "\\" + name_file
    for name_img in os.listdir(path_file):
        path_img = path_file + "\\" + name_img
        print(path_img)
        img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        list = []   # 作为 List[] 的元素
        for i in range(height):
            for j in range(width):
                list.append(img[i][j])
        List.append(list)   # 得到训练样本的数据列表

trainFaceMat = np.mat(List)  # 得到训练样本矩阵
print("\n原始样本矩阵")
print("元素个数：", trainFaceMat.shape[0])  # 400张图片
print("元素数据量：", trainFaceMat.shape[1])  # 每张图片10304像素
print(trainFaceMat)

# 平均脸矩阵
meanFaceMat = np.mean(trainFaceMat, axis=0)  # axis = 0 每一列的和除行数，得到平均值
print("\n平均脸矩阵")
print("元素个数：", meanFaceMat.shape[0])
print("元素数据量：", meanFaceMat.shape[1])
print(meanFaceMat)

# 得到减去平均值后的差值矩阵
normTrainFaceMat = trainFaceMat - meanFaceMat
print("\n差值矩阵")
print("元素个数：", normTrainFaceMat.shape[0])
print("元素数据量：", normTrainFaceMat.shape[1])
print(normTrainFaceMat)

# 求差值矩阵的协方差矩阵
covarianceMat = np.cov(normTrainFaceMat)
print("\n协方差矩阵")
print("元素个数：",normTrainFaceMat.shape[0])
print("元素数据量：",normTrainFaceMat.shape[1])
print(covarianceMat)

# 求得协方差矩阵covarianceMat的特征值eigenvalue和特征向量featurevector
eigenvalue, featurevector = np.linalg.eig(covarianceMat)
print("\n特征值：")
print(eigenvalue)
print("\n特征向量：")
print(featurevector)

# 获取 特征值按降序排序对应原矩阵 的下标
sorted_Index = np.argsort(eigenvalue)

# 保留前 30 个最大的特征值对应的特征向量
eigenvalue_bigger = featurevector[:, sorted_Index[:-30-1:-1]]


# 将 原始数据的差值矩阵 进行转置然后与 特征向量矩阵 相乘来实现降维
# 获得训练样本的特征脸空间
eigenfaceMat = np.dot(np.transpose(normTrainFaceMat), eigenvalue_bigger)

# 计算训练样本在特征脸空间的投影矩阵
eigen_train_sample = np.dot(normTrainFaceMat, eigenfaceMat)
print("\n投影样本矩阵：")
print(eigen_train_sample)

# 识别阶段
x = "D:\\desktop\\Program\\Python\\Face_Recognition\\ORL_Faces\\s18\\9.bmp"
img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
height, width = img.shape
l = []
for i in range(height):
    for j in range(width):
        l.append(img[i][j])
testFaceMat = np.mat(l)  # 得到测试图片数据矩阵
normTestFaceMat = testFaceMat - meanFaceMat
eigen_test_sample = np.dot(normTestFaceMat, eigenfaceMat)

minDistance = np.linalg.norm(eigen_train_sample[0] - eigen_test_sample)
num = 1
for i in range(1, eigen_train_sample.shape[0]):
    distance = np.linalg.norm(eigen_train_sample[i] - eigen_test_sample)
    if minDistance > distance:
        minDistance = distance
        num = i // 10 + 1
print(num)
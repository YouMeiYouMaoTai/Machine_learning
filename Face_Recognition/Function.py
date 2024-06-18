import cv2
import os
import numpy as np

Datafile_train = "D:\\desktop\\Program\\Python\\Face_Recognition\\Faces_Data\\Train_Faces"
#Datafile_train = "D:\\desktop\\train"
Datafile_test = "D:\\desktop\\Program\\Python\\Face_Recognition\\Faces_Data\\Test_Faces"
#Datafile_test = "D:\\desktop\\test"

pathlist = []   # 存放遍历的图片地址的顺序
List = []  # 存放图像数据, 元素为 列表[], 每个元素大小为 height * width

def Get_trainFaceMat():
    # 把每一张图片的地址读取出来
    for name_file in os.listdir(Datafile_train):
        path_file = Datafile_train + "\\" + name_file
        for name_img in os.listdir(path_file):
            path_img = path_file + "\\" + name_img
            pathlist.append(path_img)
            img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape
            list = []  # 作为 List[] 的元素
            for i in range(height):
                for j in range(width):
                    list.append(img[i][j])
            List.append(list)  # 得到训练样本的数据列表
    trainFaceMat = np.mat(List)  # 得到训练样本矩阵
    return trainFaceMat

trainFaceMat = Get_trainFaceMat()

def Get_meanFaceMat():
    # 平均脸矩阵
    meanFaceMat = np.mean(trainFaceMat, axis=0)  # axis = 0 每一列的和除行数，得到平均值
    return meanFaceMat

meanFaceMat = Get_meanFaceMat()

def Get_eigen_train_sample():
    # 得到减去平均值后的差值矩阵
    normTrainFaceMat = trainFaceMat - meanFaceMat
    return normTrainFaceMat

normTrainFaceMat = Get_eigen_train_sample()

def Get_eigenvalue():
    # 求差值矩阵的协方差矩阵
    covarianceMat = np.cov(normTrainFaceMat)

    # 求得协方差矩阵covarianceMat的特征值eigenvalue和特征向量featurevector
    eigenvalue, featurevector = np.linalg.eig(covarianceMat)
    return eigenvalue, featurevector

eigenvalue, featurevector = Get_eigenvalue()

def Get_sorted_Index():
    # 获取 特征值按降序排序对应原矩阵 的下标
    sorted_Index = np.argsort(eigenvalue)
    return sorted_Index

sorted_Index = Get_sorted_Index()

def Get_eigenvalue_bigger():
    # 保留前 30 个最大的特征值对应的特征向量
    eigenvalue_bigger = featurevector[:, sorted_Index[:-30 - 1:-1]]
    return eigenvalue_bigger

eigenvalue_bigger = Get_eigenvalue_bigger()

# 将 原始数据的差值矩阵 进行转置然后与 特征向量矩阵 相乘来实现降维
def Get_eigenfaceMat():
    # 获得训练样本的特征脸空间
    eigenfaceMat = np.dot(np.transpose(normTrainFaceMat), eigenvalue_bigger)
    return eigenfaceMat

eigenfaceMat = Get_eigenfaceMat()

def Get_eigen_train_sample():
    # 计算训练样本在特征脸空间的投影矩阵
    eigen_train_sample = np.dot(normTrainFaceMat, eigenfaceMat)
    return eigen_train_sample

eigen_train_sample = Get_eigen_train_sample()

# 传进 待识别的图片的地址 得到 识别结果的图片的地址
def result_img_path(path):
    ### 得到测试集的数据
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    test_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            test_list.append(img[i][j])
    testFaceMat = np.mat(test_list)  # 得到测试图片数据矩阵
    normTestFaceMat = testFaceMat - meanFaceMat     # 得到测试样本和平均脸空间的的差值矩阵
    eigen_test_sample = np.dot(normTestFaceMat, eigenfaceMat)   # 得到投影矩阵
    minDistance = np.linalg.norm(eigen_train_sample[0] - eigen_test_sample)
    num = 1
    for i in range(1, eigen_train_sample.shape[0]):
        distance = np.linalg.norm(eigen_train_sample[i] - eigen_test_sample)
        if minDistance > distance:
            minDistance = distance
            num = i
    return pathlist[num]

# 输出整体识别正确率
def success_rate():
    img_count = 0.0
    sure_count = 0.0
    for name_test in os.listdir(Datafile_test):
        img_count += 1.0
        test_num = name_test[0:-4]
        path_test = Datafile_test + "\\" + name_test
        result_path = result_img_path(path_test)
        if len(result_path) == 74:
            resulr_num = result_path[67:68]
        else:
            resulr_num = result_path[67:69]
        if resulr_num == test_num:
            sure_count += 1.0
    rate = (sure_count * 1.0 / img_count) * 100
    return rate

# 输出前十个特征空间和特征向量
def disp():
    print("排序后的前10个特征值:")
    for i in range(10, 0, -1):
        print(eigenvalue[sorted_Index[i]])
    print("排序后的前10个特征向量:")
    for i in range(10):
        print(eigenvalue_bigger[i])
    print("排序后的前10个特征空间:")
    for i in range(10):
        print(eigenfaceMat[i])

# 人脸检测函数
def Face_check(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(
        "D:/Opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 0)
    return faces

# 统计人脸检测率
def Check_rate():
    count_num = 0.0
    count_success = 0.0
    for name_file in os.listdir(Datafile_test):
        count_num += 1.0
        path_file = Datafile_test + "\\" + name_file
        if len(Face_check(path_file)):
            count_success += 1.0
    result = (count_success / count_num) * 100
    return result

import gzip
import pickle
import cv2
import numpy as np

# 加载并解压已下载文件的内容
def load_data():
    # ((training_images, training_ids),
    # (test_images, test_ids))
    mnist = gzip.open('mnist.pkl.gz', 'rb')
    training_data, test_data = pickle.load(mnist)
    mnist.close()
    return (training_data, test_data)

# 将ID转换为一个分类向量
def vectorized_result(j):
    e = np.zeros((10,), np.float32)
    e[j] = 1.0
    return e

# 重新格式化原始数据，以匹配OpenCV所期望的格式
def wrap_data():
    tr_d, te_d = load_data()
    training_inputs = tr_d[0]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_data = zip(te_d[0], te_d[1])
    return (training_data, test_data)

# 创建一个未训练的人工神经网络
def create_ann(hidden_nodes = 60):  # 使用60个隐藏结点
    ann = cv2.ml.ANN_MLP_create() # 创建一个未训练的人工神经网络
    # 配置人工神经网络的层数和节点数，例如，[784, 60, 10] 代表指定784个输入节点，10个输出节点以及包含60个节点的一个隐藏层。如果将其改成 [784, 60, 50, 10] ,将指定两个隐藏层，分别有60个节点和50个节点。
    ann.setLayerSizes(np.array([784, hidden_nodes, 50, 10]))
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0) # 激活函数
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1) # 训练方法
    ann.setTermCriteria(    # 训练终止标准
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1.0))
    return ann

# 训练函数
def train(ann, samples=50000, epochs=20):
    tr, test = wrap_data()
    tr = list(tr)
    for epoch in range(epochs):
        print("已完成 %d/%d 个阶段" % (epoch, epochs))
        counter = 0
        for img in tr:
            if (counter > samples):
                break
            if (counter % 1000 == 0): # 每处理1000个样本，输出一条样本相关信息
                print("第 %d 阶段 : 正在对第 %d/%d 个样本进行训练" % \
                      (epoch + 1, counter, samples))
            counter += 1
            sample, response = img
            data = cv2.ml.TrainData_create(
                np.array([sample], dtype=np.float32),
                cv2.ml.ROW_SAMPLE,
                np.array([response], dtype=np.float32))
            if ann.isTrained():
                ann.train(data,
                          cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
            else:
                ann.train(data, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
    print("全部完成！")
    return ann, test

# 预测函数
def predict(ann, sample):
    if sample.shape != (784,):
        if sample.shape != (28, 28):
            sample = cv2.resize(sample, (28, 28),
                                interpolation=cv2.INTER_LINEAR)
        sample = sample.reshape(784, )
    return ann.predict(np.array([sample], dtype=np.float32))

# 对给定数据集分类，测量准确度
def test(ann, test_data):
    num_tests = 0
    num_correct = 0
    for img in test_data:
        num_tests += 1
        sample, correct_digit_class = img
        digit_class = predict(ann, sample)[0]
        if digit_class == correct_digit_class:
            num_correct += 1
    print('准确率: %.2f%%' % (100.0 * num_correct / num_tests))

#ann, test_data = train(create_ann())
#test(ann, test_data)
#ann.save('mnist_ann.xml')
# myann=cv2.ml.ANN_MLP_load('mnist_ann.xml')
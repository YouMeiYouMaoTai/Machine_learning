import cv2
import numpy as np
import mnist_ann

# 框选每个数字最外层的边框
def inside(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and \
            (y1+h1 < y2+h2)

def wrap_digit(rect, img_w, img_h):
    x, y, w, h = rect

    # 长、宽 = max(长，宽)
    x_center = x + w // 2
    y_center = y + h // 2
    if (h > w):
        w = h
        x = x_center - (w // 2)
    else:
        h = w
        y = y_center - (h // 2)

    # 在四周添加 5 个像素
    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # 防止越界
    if x < 0:
        x = 0
    elif x > img_w:
        x = img_w
    if y < 0:
        y = 0
    elif y > img_h:
        y = img_h
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y

    return x, y, w, h

def predict_img(img_path):
    my_ann = cv2.ml.ANN_MLP_load('mnist_ann.xml')
    # 加载一幅测试图像
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # 图像转换成灰度图像并对其进行模糊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(gray, (7, 7), 0, gray)
    # 应用阈值以及形态学运算，确保数字从背景中脱颖而出
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    erode_kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    # 为了检测图片中的每个数字，首先需要找到轮廓
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓并找到它们的矩形框丢弃所有太大或太小而不能视为数字的矩形，
    # 同时丢弃所有完全包含在其他矩形中的矩形。把其余的矩形添加到好的矩形列表中，
    # （我们认为）这些矩形包含单个的数字
    rectangles = []
    img_h, img_w = img.shape[:2]
    img_area = img_w * img_h
    for c in contours:
        a = cv2.contourArea(c)
        if a >= 0.98 * img_area or a <= 0.0001 * img_area:
            continue
        r = cv2.boundingRect(c)
        is_inside = False
        for q in rectangles:
            if inside(r, q):
                is_inside = True
                break
        if not is_inside:
            rectangles.append(r)
    # 使用wrap_digit函数对矩阵列表进行清理，并对其内的图像数据进行分类
    for r in rectangles:
        x, y, w, h = wrap_digit(r, img_w, img_h)
        roi = thresh[y:y + h, x:x + w]
        digit_class = int(mnist_ann.predict(my_ann, roi)[0])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "%d" % digit_class, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite("result.jpg", img)

'''
predict_img("D:\\Desktop\\Program\\Python\\Machine_learn\\mnist_ann\\img_number\\99.jpg")
img = cv2.imread("D:\\Desktop\\Program\\Python\\Machine_learn\\mnist_ann\\result.jpg")
cv2.imshow("thresh", img)
cv2.waitKey()
'''
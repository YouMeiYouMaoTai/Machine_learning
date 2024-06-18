import numpy as np
from sklearn.naive_bayes import BernoulliNB
x=np.array([[0,1,0,1],
           [1,1,1,0],
           [0,1,1,0],
           [0,0,0,1],
           [0,1,1,0],
           [0,1,0,1],
           [1,0,0,1]]
          )
y = np.array([0,1,1,0,1,0,0])
#导入朴素贝叶斯
clf = BernoulliNB()
clf.fit(x,y)
#将下一天的情况输入模型
tomorrow=[[0,0,1,0]]
result=clf.predict(tomorrow)
rate=clf.predict_proba(tomorrow)
print("明天不会下雨的概率为：",rate[0][0])
print("明天真会下雨的概率为：",rate[0][1])
#输出模型预测结果
if result == 0:
    print("预测结果为：明天不会下雨")
else:
    print("预测结果为：明天真会下雨")
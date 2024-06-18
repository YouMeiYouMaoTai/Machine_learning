from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 加载数据集
# 数据集采用的是sklearn库提供的鸢尾花数据集
iris = load_iris()
X = iris.data  # 数据
y = iris.target  # 标签

# 2. 划分数据集
X_trainer, X_test, Y_trainer, Y_test = train_test_split(X, y, test_size=0.2)

# 3. RandomForestClassifier
# n_estimators：设置决策树的数量，即想要创建几棵树；数值设置得越大预测结果越稳定，但太大就会使得速度慢了；
# max_depth：设置每棵决策树的最大深度（树的层数）
clf = RandomForestClassifier(n_estimators=5, max_depth=4)
clf.fit(X_trainer, Y_trainer)
score = clf.score(X_test, Y_test)  # 模型得分,在测试集上的得分，1分最好
print("模型得分：", score)
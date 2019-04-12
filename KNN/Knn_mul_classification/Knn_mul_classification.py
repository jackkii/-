# Knn_mul_classification.py
# ----------------------------
# Knn多元分类算法
# ----------------------------
# Jackkii   2019/04/12
#


import numpy as np
# 导入数据集生成器
from sklearn.datasets import make_blobs
# 导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入画图工具
import matplotlib.pyplot as plt
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

# 生成样本数为500， 分类为5的数据集, 并赋值给X和y
data = make_blobs(n_samples=500, centers=5, random_state=8)
X1, y1 = data
# 将生成的数据可视化
plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap=plt.cm.spring, edgecolor='k')
# 图为‘生成样本’
plt.show()

# https://scikit-learn.org/dev/modules/classes.html#module-sklearn.neighbors
clf = KNeighborsClassifier()
# Fit the model using X as training data and y as target values
clf.fit(X1, y1)

# 下面代码用于画图
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
# meshgrid函数：从一个坐标向量中返回一个坐标矩阵
# meshgrid帮助文档https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
# ravel函数：将矩阵变成一个一维数组
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# pcolormesh能直观表现出散点图分类边界， 在cmap中选择颜色
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel1)
# 绘制散点图，c:色彩或颜色序列， cmap:Colormap可选，'k':black
plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap=plt.cm.spring, edgecolors='k')
# 设置坐标轴参数范围
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title('Classifier:KNN')
plt.show()

# 模型评分
print('模型正确率：{:.2f}'.format(clf.score(X1, y1)))

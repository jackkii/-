# Knn_wire_classification.py
# ----------------------------
# Knn酒的分类
# ----------------------------
# Jackkii   2019/04/13
#


import numpy as np

# 导入酒数据集
from sklearn.datasets import load_wine
# Bunch对象，包括键值(keys)和数值(values)
wine_dataset = load_wine()
print('红酒数据集中的键:\n{}'.format(wine_dataset.keys()))
# 包括'data','target','target_names',数据描述'DESCR','feature_names'
print('数据概况：{}'.format(wine_dataset['data'].shape))
# (178, 13) 178个样本，每条数据有13个变量
print(wine_dataset['DESCR'])
# 获取更详细数据信息（酒被分为3类）

# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
# 将数据集分为训练集和测试集，一般3：1【一般用大写X表示数据特征（二维数组），小写y表示数据对应标签（一维数组）】
# random_state为0或者缺省时，则每次生成的伪随机数不同
X_train, X_test, y_train, y_test = train_test_split(wine_dataset['data'], wine_dataset['target'], random_state=0)

# 导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# 用模型对数据进行拟合，返回自身
knn.fit(X_train, y_train)

# 打印模型得分
print('测试集得分：{:.2f}'.format(knn.score(X_test, y_test)))

# 新的数据点
X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
prediction = knn.predict(X_new)
print('预测新红酒分类为：{}'.format(wine_dataset['target_names'][prediction]))

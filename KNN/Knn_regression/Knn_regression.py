# Knn_regression.py
# ----------------------------
# Knn回归分析算法
# ----------------------------
# Jackkii   2019/04/12
#


import numpy as np
# 导入数据集生成器
from sklearn.datasets import make_regression
# 导入KNN分类器
from sklearn.neighbors import KNeighborsRegressor
# 导入画图工具
import matplotlib.pyplot as plt
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

# 生成特征数量为1，噪声为50的数据集
X, y = make_regression(n_features=1, n_informative=1, noise=50, random_state=8)
# 用KNN模型拟合数据
reg = KNeighborsRegressor(n_neighbors=2)         # 默认k=5时，模型正确率0.77
reg.fit(X, y)                                    # k=2时，0.86

# 预测结果图像化
z = np.linspace(-3, 3, 200).reshape(-1, 1)
plt.scatter(X, y, c='orange', edgecolors='k')
plt.plot(z, reg.predict(z), c='k', linewidth=3)
plt.title('KNN Regressor,k=2')
plt.show()

# 模型评分
print('模型正确率：{:.2f}'.format(reg.score(X, y)))

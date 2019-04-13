# linear_npoint.py
# --------------------
# 多个点拟合直线
# ---------------------
# Jackkii 2019/04/13
#

import numpy as np
import matplotlib.pyplot as plt

# 导入线性回归模型
from sklearn.linear_model import LinearRegression
# 导入数据集
from sklearn.datasets import make_regression

# 生成数据集
X, y = make_regression(n_samples=100, n_features=1, n_informative=1, noise=50, random_state=1)

# 用线性模型拟合
reg = LinearRegression()
reg.fit(X, y)

# 画出3个点和直线的图形
z = np.linspace(-3, 3, 200)
plt.scatter(X, y, c='r', s=60)
plt.plot(z, reg.predict(z.reshape(-1, 1)), c='k')
plt.title('Linear Regression')
plt.show()

print('直线方程为：y = {:.3f}'.format(reg.coef_[0]), 'x', '+{:.3f}'.format(reg.intercept_))
# 直线方程为：y = 85.145 x +10.135


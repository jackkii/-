# linear_three_point.py
# --------------------
# 3个点拟合直线
# ---------------------
# Jackkii 2019/04/13
#

import numpy as np
import matplotlib.pyplot as plt

# 导入线性回归模型
from sklearn.linear_model import LinearRegression

# 输入三个点横坐标
X =[[1], [4], [3]]
# 输入三个点纵坐标
y = [3, 5, 3]
# 用线性模型拟合3个点
lr = LinearRegression().fit(X, y)

# 画出3个点和直线的图形
z = np.linspace(0, 5, 20)
plt.scatter(X, y, s=80)
plt.plot(z, lr.predict(z.reshape(-1, 1)), c='k')
plt.title('Straight Line')
plt.show()

print('直线方程为：y = {:.3f}'.format(lr.coef_[0]), 'x', '+{:.3f}'.format(lr.intercept_))
# 直线方程为：y = 0.571 x +2.143


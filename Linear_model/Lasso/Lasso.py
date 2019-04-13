# Lasso.py
# --------------------
# 索套回归
# ---------------------
# Jackkii 2019/04/13
#

import numpy as np
import matplotlib.pyplot as plt
# 导入线性回归模型
from sklearn.linear_model import Lasso
# 导入糖尿病数据集
from sklearn.datasets import load_diabetes
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
# 导入学习曲线绘制包
from sklearn.model_selection import learning_curve, KFold

# 生成数据集
X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 降低alpha值以降低欠拟合程度(减少过多可能出现过拟合)，同时增大最大迭代次数
lasso = Lasso(alpha=0.1, max_iter=100000).fit(X_train, y_train)

# 训练得分
print('训练数据集得分：{:.2f}'.format(lasso.score(X_train, y_train)))
print('测试数据集得分：{:.2f}'.format(lasso.score(X_test, y_test)))
print('套索回归使用特征数：{}'.format(np.sum(lasso.coef_ != 0)))
# 训练数据集得分：0.36(alpha=1),0.52(alpha=0.1)
# 测试数据集得分：0.37(alpha=1),0.48(alpha=0.1)
# 套索回归使用特征数：3(alpha=1), 7(alpha=0.1)

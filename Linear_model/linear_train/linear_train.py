# linear_train.py
# --------------------
# 糖尿病线性模型预测
# ---------------------
# Jackkii 2019/04/13
#

import numpy as np
import matplotlib.pyplot as plt

# 导入线性回归模型
from sklearn.linear_model import LinearRegression
# 导入糖尿病数据集
from sklearn.datasets import load_diabetes
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

# 生成数据集
X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 用线性模型拟合
reg = LinearRegression()
reg.fit(X_train, y_train)

# 训练得分
print('训练数据集得分：{:.2f}'.format(reg.score(X_train, y_train)))
print('测试数据集得分：{:.2f}'.format(reg.score(X_test, y_test)))
# 训练数据集得分：0.53
# 测试数据集得分：0.46

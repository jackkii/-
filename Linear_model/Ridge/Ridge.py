# Ridge.py
# --------------------
# 岭回归
# ---------------------
# Jackkii 2019/04/13
#

import numpy as np
import matplotlib.pyplot as plt
# 导入线性回归模型
from sklearn.linear_model import Ridge, LinearRegression
# 导入糖尿病数据集
from sklearn.datasets import load_diabetes
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
# 导入学习曲线绘制包
from sklearn.model_selection import learning_curve, KFold

# 生成数据集
X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 用线性模型拟合,增大alpha值会降低特征变量系数（coef_)，使其趋近于0，降低训练集性能，但更有助于泛化
reg = Ridge().fit(X_train, y_train)

# 训练得分
print('训练数据集得分：{:.2f}'.format(reg.score(X_train, y_train)))
print('测试数据集得分：{:.2f}'.format(reg.score(X_test, y_test)))
# 训练数据集得分：0.43(alpha=1),0.15(alpha=10),0.52(alpha=0.1)
# 测试数据集得分：0.43(alpha=1),0.16(alpha=10),0.47(alpha=0.1)


# def plot_learning_curve(est, x, y):
#     # 将数据拆分20次对模型进行评估
#     training_set_size, train_scores, test_scores = \
#         learning_curve(est, x, y, train_sizes=np.linspace(.1, 1, 20), cv=KFold(20, shuffle=True, random_state=1))
#     estimator_name = est.__class__.__name__
#     line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label='training '+estimator_name)
#     plt.plot(training_set_size, test_scores.mean(axis=1), '-', label='test ' + estimator_name, c=line[0].get_color())
#     plt.xlabel('Training set size')
#     plt.ylabel('Score')
#     plt.ylim(0, 1.1)
#
#
# plot_learning_curve(Ridge(alpha=1), X, y)
# plot_learning_curve(LinearRegression(), X, y)
# plt.legend(loc=(0, 1.05), ncol=2, fontsize=11)

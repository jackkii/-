# Iris分类

# 导入相关库
import tensorflow as tf
import numpy as np

# 导入iris
from sklearn.datasets import load_iris
iris = load_iris()

# 查看数据集
print(iris.target)
print(iris.feature_names)
print(iris.data)
print(iris.data.shape)  #（150, 4)

# 将数据集按照3：1分为训练集和测试集
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(iris.data, iris.target, test_size=0.25)

# 打乱数据
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(iris.data, iris.target, random_state=4)

# 查看分割后数据集
print(Xtrain)
print(Xtest)
print(Ytrain)
print(Ytest)

# 定义占位符
x_train = tf.placeholder('float', [None, 4])
x_test = tf.placeholder('float', [4])

# 计算距离？
distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.negative(x_test))), axis=1)

# 获取距离最小的index
pred = tf.argmin(distance, 0)

# 定义正确率标量
accuracy = 0

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    m = len(Xtest)
    for i in range(m):
        # index[0]为最小值所在索引， index[1]为所有距离大小的列表
        index = sess.run([pred, distance], feed_dict={x_train: Xtrain, x_test: Xtest[i, :]})

        # 预测值
        pred_label = Ytrain[index[0]]

        # 真值
        true_label = Ytest[i]

        # 计算预测正确的样本数
        if pred_label == true_label:
            accuracy += 1
        print('test', i, 'predict label:', pred_label, 'true label:', true_label)

    print('accuracy:', accuracy/m)

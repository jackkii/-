# knn1.py
# -*- coding=utf-8 -*-

#导入必要的库
from numpy import *
#函数主要分为几类：对象比较、逻辑比较、算术运算和序列操作
import operator  

#创建简单数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
    
# k近邻算法
def classify(inX,dataSet,labels,k):
    # inX用于分类的输入向量 
    # dataSet输入的训练样本集
    # labels为标签向量 
    # k用于选择最近的邻居数目
    
    # 计算距离（欧式距离）
    dataSetSize = dataSet.shape[0]    
    # np.tile(A, reps) 按reps的值对A相应维度的值进行重复构建新数组，这里将inX重复dataSetSize行1列，再减去对应元素大小
    diffMat = tile(inX,(dataSetSize,1)) - dataSet     
    sqDiffMat = diffMat ** 2          # 平方
    sqDistances = sqDiffMat.sum(axis=1)     # 求和
    distances = sqDistances ** 0.5       # 开方
    sortedDistIndicies = distances.argsort()   # 排序，返回数组值从小到大的索引值
    classCount = {}
    
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 取得前k个距离最小的的点的标签
        
        # get()函数，返回字典classCount中voteilabel元素对应的值，若无，则初始化为0，若存在label则加1；
        # 在此处，则取得前k个样本的标签，存入字典并计数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) +1 
    
    # 排序,sorted可对任何序列排序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse= True) 
    # items()将dict分解为元组列表， operator.itemgetter(1)表示按照第二个元素次序对元组进行排序
    return sortedClassCount[0][0]

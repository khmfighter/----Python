#coding:utf8
from createDateSet import createDateSet
from numpy import *
import operator
#创建或导入训练集
dataSet,labels = createDateSet()
print dataSet
print labels

dataSetSize = dataSet.shape[0] #样本总数

#求出输入的新样本点到各个点的距离
inx = [0,0]
diffMax = tile(inx,(dataSetSize,1)) - dataSet
sqDiffMat = diffMax ** 2
print sqDiffMat
sqDistances = sqDiffMat.sum(axis = 1)
print sqDistances
distances = sqrt(sqDistances)
print distances
sortedDistIndicsies = distances.argsort() #数组中数值大小进行排序
print sortedDistIndicsies

classCount = {}
k = 3
#分类计数，前k个最多的
for i in range(k):
    voteIlabel = labels[sortedDistIndicsies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
print classCount
sortedClassCount = sorted(classCount.iteritems(),
                          key = operator.itemgetter(1),
                          reverse = True)

print sortedClassCount[0][0]

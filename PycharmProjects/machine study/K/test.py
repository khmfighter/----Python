#coding:utf8
import operator
from numpy import *
inx = [0,0]
dataSet = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
label=['A','A','B','B']
print dataSet

dataSetSize = dataSet.shape[0]
print dataSetSize

diffMat = tile(inx,(dataSetSize,1)) - dataSet #tile对数组array元祖进行复制
print diffMat

sqDiffMat = diffMat**2
print sqDiffMat

sqDistance = sqDiffMat.sum(axis=1)
print sqDistance

distance = sqDistance**0.5
print distance

sortDistance = distance.argsort() # 按照值得大小排列，显示出索引值
print sortDistance

classCount = {}
for i in range(3):
    voteIlabel = label[sortDistance[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #dict(key,default)
    print classCount

sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)

print sortedClassCount[0][0]
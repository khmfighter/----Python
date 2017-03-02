#coding:utf8
from numpy import *
import operator
#计算新样本与训练集之间的距离，然后返回分类结果
def classify0(inx,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inx,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedDistIndicies = sorted(classCount.iteritems(),
                                key = operator.itemgetter(1),
                                reverse=True)
    return sortedDistIndicies[0][0]
#从样本文件出分离训练集和label，转换格式

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()#每行数据划分成一个集合
    numberOfLines = len(arrayOLines) #样本个数(长度)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  #去除空格
        listFromLine = line.split('\t')  #制表符分割
        returnMat[index,:] = listFromLine[0:3] #提取前三个特征值
        classLabelVector.append(int(listFromLine[-1])) #提取label值
        index += 1
    return returnMat,classLabelVector



def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #初始化成与训练集一样规模的零矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1)) #得到归一化后的数组
    return normDataSet,ranges,minVals



#验证分类器

def datingClassTest():
    hoRatio = 0.1 #提取样本中百分之十做测试集
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],
                                     normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],
                                     3)
        print "the classifier came back with : %d and real %d" % (classifierResult,datingLabels[i])

        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is %f" % (errorCount/float(numTestVecs))


def classifyPerson():
    resultList = ['not at all','is small doses','in large doses']
    percenTats = float(raw_input("frequent flier miles earned per year?"))
    ffMiles = float(raw_input("liters of ice cream consumed per year?"))
    icecream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat) #训练集归一化处理
    inArr = array([ffMiles,percenTats,icecream])
    classifierResult = classify0((inArr-minVals)/ranges,
                                 normMat,datingLabels,3) #分类核心算法
    print "就分类结果是" + resultList[classifierResult - 1]


classifyPerson()

#coding:utf8
from numpy import  *

from adabootTool import *

def loadData(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat , labelMat

datArr,labelArr = loadData("horseColicTraining2.txt")
classifierArr = adaBoostTrainDs(datArr,labelArr,10)
#print classifierArr

testArr,labelArr = loadData('horseColicTest2.txt')
prediction10 = adaClassify(testArr,classifierArr)
print prediction10

#coding:utf8
from numpy import *


#读取数据，处理数据
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#从中某个区间随机选取一个数
def selectJrand(i,m):
    j=i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

#调整值太大的数
def clipAlpha(aj,H,L):
    if aj>H:
        aj = H
    if aj<L:
        aj = L
    return aj


dataMat,labelMat = loadDataSet('testSet.txt')
print dataMat
print labelMat


def smoSimple(dataMat,classLabels,C,toler,maxIter):

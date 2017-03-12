#coding:utf8
from numpy import *

#从文本读取的数据都为字符串类型，需要转换成浮点型
#加载原始数据
def loadData(fileName):
    fr = open(fileName)
    dataMat = []
    labelMat = []
    numFeat = len(fr.readline().split('\t')) - 1
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular ,cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


'''
#matplotlib依赖库没解决
dataMat,labelMat = loadData("ex0.txt")

ws = standRegres(dataMat,labelMat)

xMat = mat(dataMat)
yMat = mat(labelMat)
yHat = xMat * ws
#print yHat

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])


'''

#局部加权线性回归（最小二乘法），解决欠拟合情况
def lwlr(testPoint,xArr,yArr,k = 1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular ,cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

#求局部带权线性回归，返回测试集合
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat




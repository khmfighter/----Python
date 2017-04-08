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




#print lwlr(xArr[0],xArr,yArr,k=1.0)
#print lwlrTest(xArr,xArr,yArr,k=1.0)


#岭回归，当特征值大于样例数时，解决矩阵非奇异问题
def ridgeRegression(xMat,yMat,lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular , cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    #均值归一化
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegression(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def rssError(yArr,yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


#xArr,yArr = loadData("abalone.txt")
#ridgeWeights = ridgeTest(xArr,yArr)
#print ridgeWeights


#前向逐步回归
def stageWise(xArr,yArr,eps=0.01,numIter=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMean = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMean)/xVar
    m,n = shape(xMat)
    returnMat = zeros((numIter,n))
    ws = zeros((n,1))   #创造向量来保存weights的值
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIter):
        print ws.T
        lowerstError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowerstError:
                    lowerstError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat
xArr,yArr = loadData("abalone.txt")

print stageWise(xArr,yArr,0.01,200)[0:10,:]
print stageWise(xArr,yArr,0.001,5000)[0:10,:]

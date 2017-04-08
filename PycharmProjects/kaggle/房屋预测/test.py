#coding:utf8
from numpy import *
from numpy import argmin
import pandas as pd

feature = ['GarageCars','GrLivArea','TotRmsAbvGrd']

trainSet = pd.read_csv('train.csv')
testSet = pd.read_csv('test.csv')

label = trainSet['SalePrice']

testSet_GarageCars = testSet[feature[0]]
testSet_GrLivArea = testSet[feature[1]]
testSet_TotRmsAbvGrd = testSet[feature[2]]

feature_GarageCars = trainSet[feature[0]]
feature_GrLivArea = trainSet[feature[1]]
feature_TotRmsAbvGrd = trainSet[feature[2]]

all_data = pd.concat([feature_GarageCars,feature_GrLivArea,feature_TotRmsAbvGrd],axis=1)
testall_data = pd.concat([testSet_GarageCars,testSet_GrLivArea,testSet_TotRmsAbvGrd],axis=1)



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
ws = standRegres(all_data,label)
SalePrice = mat(testall_data) * ws
print SalePrice[:100]
'''


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
#取前1000个样例作为训练集，后460个作为测试集
#训练集处理
train_set =  mat(all_data)[:1000]
train_set_label = mat(trainSet['SalePrice'][:1000])

#测试集处理
test_set = mat(all_data)[1000:]
test_set_label = mat(trainSet['SalePrice'][1000:])

wMat = ridgeTest(train_set,train_set_label)

result = []
for i in wMat:
    result.extend(mat(i*test_set.T))

wMat_result = []
for i in result:
    result_test = (abs(i -test_set_label)).sum()
    wMat_result.append(result_test)


minIndex = mat(wMat_result).argmin()
wMat_min =mat(wMat[17])

print (wMat_min * mat(testall_data).T)[:100]


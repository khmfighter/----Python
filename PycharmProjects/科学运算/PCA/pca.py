#coding:utf8
from numpy import *

def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    detArr = [map(float,line) for line in stringArr]
    return mat(detArr)




def pca(dataMat,topNfeat = 9999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    #协方差矩阵
    covMat = cov(meanRemoved,rowvar = 0)
    #奇异值分解，求协方差举证特征值和特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))
    #对特征值进行排序
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    #去除前k列
    redEigVects = eigVects[:,eigValInd]
    lowDataMat = meanRemoved * redEigVects
    # 特征向量和协方差矩阵相乘，得到新的降维后的矩阵
    reconMat = (lowDataMat * redEigVects.T) + meanVals
    return lowDataMat,reconMat


'''
dataMat = loadDataSet('testSet.txt')

lowDMat,reconMat = pca(dataMat,1)
print lowDMat
print reconMat
print shape(lowDMat)
'''

'''
dataMat = loadDataSet('testSet.txt')
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
print ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],
                 maker='^',s=90)
'''
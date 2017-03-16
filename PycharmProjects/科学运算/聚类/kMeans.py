#coding:utf8
from numpy import *

#加载数据
def loadData(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

#计算距离
def distEclud(VecA,VecB):
    return sqrt(sum(power(VecA - VecB,2)))

#计算质心,随机质心选取，k为质心数量
def randCent(dataSet , k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids




#print dataMat

def kMean(dataSet,k,distMean = distEclud , createCent = randCent):

    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMean(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis = 0)
    return centroids,clusterAssment

'''
dataMat = mat(loadData('testSet.txt'))
print randCent(dataMat,2)
centroids,clusterAssment = kMean(dataMat,4)
print centroids
'''


#二分k均值聚类算法，将一个质点分成多个质点
def bikmeans(dataSet,k,distMeans = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    #将每个样本与质点的误差存入clusterAssment中
    for j in range(m):
        clusterAssment[j,1] = distMeans(mat(centroid0),dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroid0,splitCluAssment = kMean(ptsInCurCluster,2)
            sseSplit = sum(splitCluAssment[:,1])
            sseNotSplit = sum(sum)
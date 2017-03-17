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
    #用矩阵存储样例所属簇，和与质心之间的距离平方差
    clusterAssment = mat(zeros((m, 2)))
    #初始化质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    #用列表保留所有质心
    centList = [centroid0]
    #将每个样本与质点的误差存入clusterAssment中
    for j in range(m):
        clusterAssment[j,1] = distMeans(mat(centroid0),dataSet[j,:])**2
    #不断进行循环直到质心的数量达到k为止
    while (len(centList) < k):
        lowestSSE = inf #正无穷
        for i in range(len(centList)):
            #提取出质心为了原来质心的样本
            ptsInCurCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat,splitClustAss = kMean(ptsInCurCluster,2)
            sseSplit = sum(splitClustAss[:,1])
            #提取出新划分的质心样本
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            #划分和没被划分的样本距离质心的平方差
            print 'sseSplit , and notSplit: ',sseSplit,sseNotSplit
            #找到更小的划分索引和划分质心，使样例到质心的距离更小
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is :' , bestCentToSplit
        print 'the len of bestClustAss is :',len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment



dataMat3 = mat(loadData('testSet2.txt'))
centList , myNewAssment = bikmeans(dataMat3,3)
for i in centList:
    print i


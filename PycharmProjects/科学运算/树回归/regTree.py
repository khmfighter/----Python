#coding:utf8
from numpy import *

#读取数据
def loadData(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

#对数据进行划分
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

#生成叶子节点，取均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

#误差估算函数
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#选择切分最小误差的特征
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    beatS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:, featIndex].T.tolist()[0])):
        #对样本进行切分
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            #判断是否符合最小样本数
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):continue
            newS = errType(mat0) + errType(mat1)
            if newS < beatS:
                bestIndex = featIndex
                bestValue = splitVal
                beatS = newS
    if (S - beatS) < tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue

#创建树的节点，递归算法
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


myDat = loadData('ex00.txt')
myMat = mat(myDat)
print createTree(myMat)



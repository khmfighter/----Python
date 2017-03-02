#coding:utf8
from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCount = {}
    #最后一类类别数量统计
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    #计算不同类别香浓值
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#dataSet[1][-1] = 'maybe'
#print calcShannonEnt(dataSet)

#按照规定特征划分数据集
def splitDataSet(dataSet,axis,values):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == values:
            reducedFeatVec = featVec[:axis]
#以元素的方式加到列表中
            reducedFeatVec.extend(featVec[axis+1:])
#append以集合的方式加到列表中
            retDataSet.append(reducedFeatVec)
    return retDataSet



#dataSet,labels = createDataSet()
#print splitDataSet(dataSet,1,1)
#print splitDataSet(dataSet,2,'yes')
#print splitDataSet(dataSet,2,'no')


#选择最好的数据集划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniquVals = set(featList)
        newEntropy = 0.0
        for value in uniquVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


dataSet,labels = createDataSet()

print chooseBestFeatureToSplit(dataSet)


#分类代码
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)

    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value)
                                                  ,subLabels)

    return myTree


def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

    return classLabels


#数据存储
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

myDat , labels = createDataSet()
myTree = createTree(myDat,labels)
storeTree(myTree,'classifierStorage.txt')
print grabTree('classifierStorage.txt')

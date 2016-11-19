from math import log
import operator
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]  = 1
        else:
            labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntries)
        shannonEnt  -= prob * log(prob,2)

    return shannonEnt

def spliteData(dataSet,axis,value):
    resultData = []
    for featVec in resultData:
        if featVec[axis] == value:
            reduceData = featVec[:axis]
            reduceData.extend(featVec[axis+1:])
            resultData.append(reduceData)

    return resultData

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],
               [0,0,'no']]
    labels = ['no surfing','flippers']
    return dataSet,labels

dataSet,labels = createDataSet()
s = calcShannonEnt(dataSet)
print s


def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in numFeature:
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = spliteData(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    index = {}
    for feat in classList:
        if feat not in index.keys():
            index[feat] = 0
        index[feat] += 1
    sortList = sorted(index.iteritems(),key=operator.itemgetter(1),reverse=True)

    return sortList[0][0]

def createTree(dataSet ,Labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == dataSet[0]:
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLebels = Labels[bestFeat]
    myTree = { bestFeat:{}}
    del(Labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    for value in featValues:
        subLabels = Labels[:]
        myTree [bestFeatLebels][value] = createTree(spliteData(dataSet,bestFeat,value),subLabels)
    return myTree




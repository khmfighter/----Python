#coding:utf8
from numpy import *

#数据预处理，缺失值处理


#分类函数
def classify(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5 :
        return 1.0
    else:
        return 0.0

#sigmoid函数计算
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#改进的随机梯度上升算法，alpha值更新
def stoGrabAscent1(dataMatrix,classLabel,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = float(classLabel[randIndex]) - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
        return weights

#用训练集训练学习机，用测试集测试错误率，返回错误率
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabel = []
    for line in frTrain.readlines():
        currenLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currenLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(currenLine[-1])
    trainWeigts = stoGrabAscent1(array(trainingSet),trainingLabel,500)
    errorCount = 0.0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if classify(array(lineArr),trainWeigts) != trainingLabel[i]:
            errorCount += 1.0

    errorRate = (float(errorCount))/numTestVec
    print "错误率是 %d" % errorRate
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "十次迭代之后的错误率为：%d" % (float(errorSum)/float(numTests))
    return errorSum


multiTest()


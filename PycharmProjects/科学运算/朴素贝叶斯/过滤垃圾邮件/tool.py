#coding:utf8
from numpy import *


#人工手动构建数据集
def loadDataSet():
    postingList = [
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]

    classVec = [0,1,0,1,0,1]
    return postingList,classVec

#接收数据集，除重，list化,制作词典
def createVocaList(dataSet):
    vocaSet = set([])
    for document in dataSet:
        vocaSet = vocaSet | set(document)

    return list(vocaSet)


#输入变量，查找词典,词集性模型
def setOfWords2Vec(vocaList,inputSet):
    returnVec = [0]*len(vocaList)
    for word in inputSet:
        if word in vocaList:
            returnVec[vocaList.index(word)] = 1
        else:
            print "the word : %s is not in my Vocabulary" % word
    return returnVec

#词袋性模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#分类，训练文档分类器，朴素贝叶斯的训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  #词条数量
    numWords = len(trainMatrix[0])    #
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom

    return p0Vect,p1Vect,pAbusive



#分类，训练文档分类器，朴素贝叶斯的训练函数2.0
def trainNB2(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  #词条数量
    numWords = len(trainMatrix[0])    #
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify,p0Vect,p1Vect,pClass):
    p11 = vec2Classify * p1Vect
    p00 = vec2Classify * p0Vect
    print p11
    print p00
    p1 = sum(p11) + log(pClass)
    p0 = sum(p00) + log(pClass)

    print p1
    print p0
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVotabList = createVocaList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVotabList,postinDoc))
    p0V,p1V,pAb = trainNB2(array(trainMat),array(listClasses))
    print p0V
    print p1V
    #新样本数据
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVotabList,testEntry))
    print testEntry,"classified as:",classifyNB(thisDoc,p0V,p1V,pAb)

    #新样本数据
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVotabList,testEntry))
    print testEntry,"classified as :",classifyNB(thisDoc,p0V,p1V,pAb)










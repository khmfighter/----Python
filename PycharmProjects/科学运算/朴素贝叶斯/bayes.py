#coding:utf8


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


#输入变量，查找词典
def setOfWords2Vec(vocaList,inputSet):
    returnVec = [0]*len(vocaList)
    for word in inputSet:
        if word in vocaList:
            returnVec[vocaList.index(word)] = 1
        else:
            print "the word : %s is not in my Vocabulary" % word
    return returnVec


listPost, postLabel = loadDataSet()

print listPost
print postLabel

myVocaList = createVocaList(listPost)
print myVocaList

print setOfWords2Vec(myVocaList,listPost[0])
print setOfWords2Vec(myVocaList,listPost[3])
print setOfWords2Vec(myVocaList,listPost[5])


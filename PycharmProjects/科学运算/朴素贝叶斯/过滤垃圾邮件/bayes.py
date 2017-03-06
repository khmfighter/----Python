#coding:utf8

import tool
from random import *
from numpy import *
#空格划分字符串
mySent = 'This book is the best book on Python or Mr.L I have ever laid eyes upon'
print mySent.split()

#标点符号划分
import re
regEx = re.compile('\\W*')
listOfTokens = regEx.split(mySent)
print listOfTokens
print [tok.lower() for tok in listOfTokens if len(tok) > 0]


file = open('email/ham/6.txt').read()
listOfTokens = regEx.split(file)
print listOfTokens

#处理字符，大于两个字母长度，分割字符
def textParse(bigString):
    import re
    listOfTokens = re.split('/W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


#处理垃圾邮件
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = tool.createVocaList(docList)
    trainningSet = range(50)
    testSet = []
    for i in range(10):
        #随机选择0-49中的十个数字，即从50封邮件里随机选取10封邮件
        randIndex = int(random.uniform(0,len(trainningSet)))
        testSet.append(trainningSet[randIndex])
        del(trainningSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainningSet:
        trainMat.append(tool.setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0v,p1v,pSpam = tool.trainNB2(array(trainMat),array(trainClasses))
    errorCount = 0.0
    for docIndex in testSet:
        wordVector = tool.setOfWords2Vec(vocabList,docList[docIndex])
        if tool.classifyNB(array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the errorCount is %d ' % errorCount

spamTest()
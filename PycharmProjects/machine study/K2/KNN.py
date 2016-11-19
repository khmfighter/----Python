#coding:utf8

from numpy import *
def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    file = open(filename)
    arrayOLines= file.readlines()
    numberOfLine = len(arrayOLines)
    returnMat = zeros((numberOfLine,3))
    classLabelVector = []
    index = 0
    for line in  arrayOLines:
        line = line.strip() #删除空白格
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index = index + 1
    return classLabelVector,returnMat


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normData = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normData = dataSet - tile(minVals,(m,1))
    normData = normData/tile(ranges,(m,1))
    return normData,ranges,minVals

array1,array2 = file2matrix('datingTestSet.txt')
normData1,ranges,minVals = autoNorm(array2)
print normData1,minVals,ranges
#print array1
#print array2


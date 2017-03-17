#coding:utf8

from numpy import *
def loadDataSet():
    datMat = matrix([
        [1.0,2.1],
        [2.0,1.1],
        [1.3,1.0],
        [1.0,1.0],
        [2.0,1.0]]
    )
    classLabel = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabel

def stumClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
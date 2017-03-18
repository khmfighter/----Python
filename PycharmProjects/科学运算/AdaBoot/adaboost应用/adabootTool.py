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

#给定阈值，进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray


#构建一颗最有单层决策树
def buildStump(dataArr,classLabel,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabel).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                #print "split : dim %d,thresh %.2f ,thresh ineqal %s,the weighted" \
                #      "error  is %.3f" % (i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

'''
datMat,classLabel = loadDataSet()
D = mat(ones((shape(datMat)[0],1))/shape(datMat)[0])
dataMat,minError,bestClassEst = buildStump(datMat,classLabel,D)
print dataMat,minError,bestClassEst
'''

def adaBoostTrainDs(dataArr,classLabels,numIter = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIter):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #print 'D:',D
        alpha = float(0.5*log((1.0 - error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print "classEst:",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        #print "aggClassEst :",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error:",errorRate,"\n"
        if errorRate == 0.0:
            break

    return weakClassArr




def adaClassify(daToClass,classifierArr):
    dataMatrix = mat(daToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print "aggClassEst:",aggClassEst
    return sign(aggClassEst)

'''
datMat,classLabel = loadDataSet()
classifierArray = adaBoostTrainDs(datMat,classLabel,9)
print adaClassify([0,0],classifierArray)
print adaClassify([[5,5],[0,0]],classifierArray)
print sign(0.69314718+1.66610226+2.56198199)
'''
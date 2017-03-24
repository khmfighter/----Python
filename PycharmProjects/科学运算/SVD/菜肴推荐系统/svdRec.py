#coding:utf8
from numpy import *
from tool import *

def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 :
            continue
        overlap = nonzero(logical_and(dataMat[:,item].A>0,
                                      dataMat[:,j].A>0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overlap,item],
                                 dataMat[overlap,j])
        print 'the %d and %d similarity is : %f ' % (item,j,similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]
    if len(unratedItems) == 0:return 'you rated everything'
    itemScore = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        itemScore.append((item,estimatedScore))
    return sorted(itemScore,
                  key = lambda jj:jj[1],reverse = True)[:N]

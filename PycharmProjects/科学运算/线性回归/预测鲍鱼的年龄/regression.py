#coding:utf8
from numpy import *
from tool import *

def rssError(yArr,yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

abX,abY = loadData("abalone.txt")
#print abX
#print abY

#yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
#print rssError(abY[0:99],yHat01)

#yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
#print rssError(abY[0:99],yHat1)

#yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
#print rssError(abY[0:99],yHat10)


yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
print rssError(abY[100:199],yHat01.T)

yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
print rssError(abY[100:199],yHat1.T)

yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
print rssError(abY[100:199],yHat10.T)

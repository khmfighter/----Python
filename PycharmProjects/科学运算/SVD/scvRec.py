#coding:utf8
from numpy import *

def loadData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

Data = loadData()
U,sigma,VT = linalg.svd(Data)

Sig3 = mat([[sigma[0],0,0],[0,sigma[1],0],[0,0,sigma[2]]])
print shape(U[:,:3]*Sig3*VT[:3,:])
print Data - U[:,:3]*Sig3*VT[:3,:] #重构与原始矩阵近似的矩阵

from numpy import linalg
#几种相似度的计算
#欧几里得距离
def eulidSim(inA,inB):
    return 1.0/(1.0 + linalg.norm(inA - inB))
#皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA) < 3 :return 1.0
    return 0.5 + 0.5 * corrcoef(inA,inB,rowvar=0)[0][1]
#余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = linalg.norm(inA)*linalg.norm(inB)
    return 0.5+0.5*(num/denom)


myData = mat(loadData())
#print myData
print eulidSim(myData[:,0],myData[:,4])
print eulidSim(myData[:,0],myData[:,0])

print pearsSim(myData[:,0],myData[:,4])
print pearsSim(myData[:,0],myData[:,0])

print cosSim(myData[:,0],myData[:,4])
print cosSim(myData[:,0],myData[:,0])
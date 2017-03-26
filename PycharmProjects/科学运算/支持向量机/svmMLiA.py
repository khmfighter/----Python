#coding:utf8
from numpy import *


#读取数据，处理数据
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#从中某个区间随机选取一个数
def selectJrand(i,m):
    j=i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

#调整值太大的数
def clipAlpha(aj,H,L):
    if aj>H:
        aj = H
    if aj<L:
        aj = L
    return aj




#maxIter表示运行次数，最大循环数,toler为间隔，常数C为
def smoSimple(dataMat,classLabels,C,toler,maxIter):
    dataMat = mat(dataMat)
    labelMat = mat(classLabels).transpose()
    b = 0
    m,n = shape(dataMat)
    alphas = mat(zeros((m,1)))
    iter = 0
    #alpha优化
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #fXi为预测的类别
            fXi = float(multiply(alphas,labelMat).T*\
            (dataMat*dataMat[i,:].T)) + b
            #预测结果与真实值之间的差值
            Ei = fXi - float(labelMat[i])
            #判断该alpha是否可以被优化，C为松弛变量的值，alpha大于零小于C
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] *Ei > toler) and (alphas[i] > 0)):
                #选定一个alpha，从剩下里面随机选取另一个alpha
                j = selectJrand(i,m)
                #选取的另一个alpha值预测真实值与预测值之间的差值
                fXj = float(multiply(alphas,labelMat).T*\
                            (dataMat*dataMat[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy() #深拷贝
                alphaJold = alphas[j].copy() #浅拷贝
                #如果alphai 与 alphaj 异号，则：
                if(labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C+alphas[j] - alphas[i])
                #如果同号，则：（ 只有+1 或 -1两种取值）
                else:
                    L = max(0,alphas[i] + alphas[j] - C)
                    H = min(C,alphas[i]+alphas[j])
                if L == H:
                    #print "L == H"
                    continue
                eta = 2.0 * dataMat[i,:]*dataMat[j,:].T - \
                    dataMat[i,:]*dataMat[j,:].T - \
                    dataMat[j,:]*dataMat[j,:].T
                if eta >= 0 :
                    #print "eta>=0"
                    continue
                #对alphaj进行改变，与alpha改变相反，方向改变，一大一小，一加一减
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #检查alphaj是否有改变
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    #print "J not moving enough"
                    continue
                #对alphai进行改变
                alphas[i] += labelMat[j]*labelMat[i]*\
                             (alphaJold - alphas[j])

                #为
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMat[i,:]*dataMat[i,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMat[i,:]*dataMat[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMat[i,:]*dataMat[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMat[j,:]*dataMat[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                #print "iter: %d i : %d,pairs changed %d" %\
                      #(iter,i,alphaPairsChanged)
        if(alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        #print "iteration number : %d" % iter

    return b,alphas


dataMat,labelMat = loadDataSet('testSet.txt')
#print dataMat
#print labelMat


b ,alpha = smoSimple(dataMat,labelMat,0.6,0.001,40)
print alpha
print b


#coding:utf8



dataMat = []
with open('train.csv') as fr:
    for line in fr.readlines():
        lineArr = line.strip(',').split('\t')
        for x in lineArr:
            lineArr.append(x)

print dataMat[0]

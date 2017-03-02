#coding:utf8
import tree


#递归计算决策树节点数
def getNumLeaf(myTree):
    numLeaf = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeaf += getNumLeaf(secondDict[key])
        else :
            numLeaf += 1
    return numLeaf


myDat,labels = tree.createDataSet()
print myDat,labels
myTree = tree.createTree(myDat,labels)
print myTree

print getNumLeaf(myTree)

#计算决策树层数

def getTreeDept(myTree):
    maxDept = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDept = 1 + getTreeDept(secondDict[key])
        else:
            thisDept = 1
        if thisDept > maxDept :
            maxDept = thisDept

    return maxDept

print getTreeDept(myTree)
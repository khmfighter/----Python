#coding:utf8
from numpy import *
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}


    def inc(self,numOccur):
        self.count += numOccur

    def disp(self,ind = 1):
        print ' '*ind,self.name,' ',self.count
        for child in self.children.values():
            child.disp(ind+1)




def createTree(dataSet,minSup = 1):
    #头指针，存储所有头指针，频繁项集元素出现个数
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            #headerTable.get(item,0)如果字典内有item项则返回值，否则返回0
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]
    for k in headerTable.keys():
        if headerTable[k] < minSup:
             del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None,None
    #？？？？？？
    for k in headerTable:
        headerTable[k] = [headerTable[k],None]
    retTree = treeNode('Node Set',1,None)
    print dataSet.items()
    #对某个项集内的数进行排序
    for tranSet , count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                    localD[item] = headerTable[item][0]
        if len(localD) > 0:
            #对一个元素项内的元素进行排序处理，选出当前出现频率最高的那个
            orderedItems = [v[0] for v in sorted(localD.items(),
                                                     key = lambda  p:p[1],
                                                     reverse = True
                                                     )]
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree , headerTable

def updateTree(items,inTree,headerTable,count):
    #判断元素项第一个数是否为该树的节点
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    #若不是，则将该节点做为树的孩子，添加，为该节点创建新的树结构
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree)

        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])

    if len(items) > 1:
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)

def updateHeader(nodeToTest,targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

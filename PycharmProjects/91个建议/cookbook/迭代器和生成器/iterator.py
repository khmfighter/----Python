#coding:utf8
#简单的迭代
items = [1,2,3]

it = iter(items)
print next(it)
print next(it)
print next(it)


#利拥内部函数__iter__()来实现迭代
class Node():
    def __init__(self,value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_children(self,children):
        self._children.append(children)

    def __iter__(self):
        return iter(self._children)

root = Node(0)
a = Node(1)
b = Node(2)
print a,b
root.add_children(a)
root.add_children(b)

for r in root:
    print r

#迭代所有组合排列
from itertools import permutations
a = ['a','b','c']
for p in permutations(a):
    print p
for p in permutations(a , 2):
    print p
for p in permutations(a , 1):
    print p

from itertools import combinations

for c in combinations(a,2):
    print c

for c in combinations(a,3):
    print c

#索引值迭代
my_list = ['a','b','c']
for index,value in enumerate(my_list):
    print index,value

for index,value in enumerate(my_list,1):
    print index,value

#同时迭代多个序列

x = [1,2,3,4,5,6,7,8,9,0]
y = [0,9,8,7,6,5,4,3,2,1]

for x , y in zip(x,y):
    print (x,y)

#from itertools import izip_longest

#for i in izip_longest(x,y):
#   print i
a = [1,2,3]
b = [4,5,6]
c = [8,9,10]

for i in zip(a,b,c):
    print i

#不同容器的迭代
from itertools import chain
for x in chain(a,b,c):
    print x

import heapq

a = [5,2,7,9]
b = [4,6,8,3]
for c in heapq.merge(a,b):
    print c
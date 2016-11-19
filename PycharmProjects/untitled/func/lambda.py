add = lambda x,y:x+y

print add(1,2)

add = lambda x,y=2:x+y
print add(3)

add = lambda x,y=2:x+y
print add(3,5)

add = lambda x,y=2:x+y
print add(0)

add = lambda x,y=2:x+y
print add(0,9)

import random

def add(n):
    return n%2

allNum=[]
a=0
for each in range(9):
    a = randint(1,99)
    print a
    allNum.append(a)

print filter(add,allNum)


allNum=[]
for each in range(9):
    allNum.append(randint(1,99))
print [ n for n in allNum if n%2]

print map(lambda x:x**2,[0,1,2,3,4,5,6])
print map(lambda x:x+2,[0,1,2,3,4,5,6])
print [n+2 for n in range(6)]
print [n**2 for n in range(6)]
print map(lambda x, y: x + y, [1,3,5], [2,4,6])
print map(lambda x, y: (x+y, x-y), [1,3,5], [2,4,6])
print map(None, [1,3,5], [2,4,6])
print zip([1,3,5],[2,4,6])
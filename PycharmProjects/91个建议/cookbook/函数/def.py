#coding:utf8
#接受任意数量参数的参数
def any(hello1,*hello):
    for i in hello:
        print i
    print hello1
s = [1,2,3,4,5,6,78,8,9]
any(1,s)
any(1,2,3,4,5,6,7,8,9,9)

def some(hello,*hello1,**hello2):
    for key,value in hello2.iteritems():
        print key + value

    print hello
    print hello1

some(1,2,3,4,5,6,x="1",y="2",z="3")

#编写可接受关键字参数的函数,pyhon3.0特性

'''def recv(hello,*,world):
    print world
    print hello

recv(1,world=2)'''

'''def minM(*value,hello=None):
    m = min(value)
    if hello is not None:
        m = hello if hello > m else m
    return m
print minmin(11,2,3,54,56,0,1)
print min(11,2,3,54,56,0,hello=1)'''

#函数附加信息元信息
'''def add(x:int,y:int) ->int:
    return x + y

add1 = add(1,2)'''


#定义匿名或者内联函数

s = lambda x,y : x + y
print s(1,2)
print s('hello','world')

#在匿名函数中绑定变量的值
x = 10
a = lambda y : x + y
x = 20
b = lambda y :x + y
print b(10)
print a(10)


a = lambda y,x=10: x + y
b = lambda y,x=20: x + y
print a(10)
print b(10)

funcs = [lambda x: x+n for n in range(5)]
for f in funcs:
    print f(0)

funcs = [lambda x,n=n : x+n for n in range(5)]
for f in funcs:
    print f(0)


#带有N个对象的可调用对象以较少的参数形式调用
def spam(a,b,c,d):
    print a,b,c,d

from functools import partial

s1 = partial(spam,1)
print s1(1,2,34)
print s1(4,5,6)
s2 = partial(spam,d = 2)
print s2(1,2,3)
s3 = partial(spam,1,2,d = 3)
print s3(1)
print s3(3)

points = [(1,2),(2,3),(3,4),(4,5)]
import math
def distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    print math.hypot(x2-x1,y2-y1)
    return math.hypot(x2-x1,y2-y1)

pt = (4,3)
points.sort(key = partial(distance,pt))

print points

#访问闭包内的变量
def sample():
    n = 0
    def func():
        print 'n = ',n

    def set_n(value):
#        nonlocal n 使用nonlocal声明是的编写函数来修改内层变量成为可能
        n = value

    def get_n():
        return n
    func.get_n = get_n
    func.set_n = set_n

    return func

f = sample()
f.set_n(10)
print f.get_n()




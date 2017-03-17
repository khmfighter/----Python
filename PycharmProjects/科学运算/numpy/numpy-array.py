#coding:utf8
from numpy import array
from numpy import arange
from numpy import hstack,vstack
from numpy import hsplit,vsplit


a = array([[1,2],[2,3]])
print a
print a.shape

b = array([[1,2,3,4],[5,6,7,8]])
print b
print b.shape

print a[0,0],a[0,1]
print b[1,2],b[:,:],b[:,2:3],b[:1,:1],b[:,:1]

a = arange(9)
print a,type(a)
print a[3:7]
print a[0:7:2]
print a[0:9:3]
print a[::-1]

b = arange(24).reshape(2,3,4)
print b
print b.shape
print b[0,...]
print b.ravel()
b.shape = (6,4)
print b.transpose()



c = arange(9).reshape(3,3)
print c
print 2*c
print c ** 2

d = 2*c

print hstack((c,d))
print vstack((c,d))


a = arange(9).reshape(3,3)
print a
print hsplit(a,3)
print vsplit(a,3)


#单位矩阵
from numpy import eye
a = eye(2)
b = eye(3)
print a
print b

#存储文件
from numpy import savetxt

savetxt('hello.txt',a*2)


#ndarray
from numpy import clip,compress

a = arange(5)
print a
print a.clip(1,2)
print a.compress(a>2)

#阶乘

b = arange(1,9)
print b
print b.prod()
print b.cumprod()


a = arange(15).reshape(3,5)
print a
print a.shape
print a.ndim
print a.dtype
print a.itemsize
print a.size
#创建数组

a = array([1,2,3])
print a
b = array([(1,2,3),(4,5,6)])
print b
#创建零元素
from numpy import *
print zeros((3,4))
print ones((4,4))
print empty((2,3))

#矩阵乘法
A = array([[1,1],[1,0]])
B = array([[2,1],[2,1]])

print dot(A,B)
print A*B

#对应元素相乘

a = ones((2,3),dtype = int)
b = random.random((2,3))
a *= 3
print a
b += a
print b


#axis的应用

a = arange(12).reshape(3,4)
print a
print a.sum(axis=0)
print a.min(axis=0)
print a.min(axis=1)
print a.cumsum(axis=1)

#通用函数

B = arange(3)
A = arange(3)
print B
print exp(B)
print sin(B)
print sqrt(B)
print cos(B)
print add(A,B)

#ravel用法

a = arange(12).reshape(3,4)
print a
print a.shape
print a.ravel()
print a.ravel().shape
a.shape = (6,2)
print a
print a.transpose()

a.resize((2,6))
print a

#通过数组索引
a = arange(12) ** 2
i = array([1,1,3,8,5])
print a
print a[i]
j = array(
    [[1,2],[3,4]]
)
print a[j]

a = arange(12).reshape(3,4)
print a
i = array([[1,2],[0,1]])
j = array([[0,1],[3,2]])
print a[i,j]


a = arange(5)
print a
a[[1,3,4]] = 0
print a
a = arange(5)
a[[0,0,2]] = [1,2,3]
print a
a[[0,1,2]] += 1
print a

a = arange(12).reshape(3,4)
print a
b = a > 4
print b
print a[b]

a[b] = 0

print a
a = array([1,2,3,4])
b = array([8,4,5])
c = array([5,4,6,8,3])

ax,bx,cx = ix_(a,b,c)
print ax
print bx
print cx
print ax.shape,bx.shape,cx.shape
print ax+bx*cx


u = arange(12).reshape(3,4)
print u.trace()
a = eye(5)
print a
print u.transpose()
print u.transpose().trace()
print trace(u)
print trace(a)

#矩阵类

A = matrix('1.0 2.0 ; 3.0 4.0')
print A
print A.T

x = matrix('7.0 9.0')
print x
y = x.T
print A * y
print (A*y).I

print 'hello'

print random.rand(4,4)
print random.rand(10)


a = arange(4).reshape(2,2)
b = [0,0]
print a
print tile(b,(4,1)) #创建0数组

print a.argsort() #数组中数值大小进行排序

from random import uniform

print int(random.uniform(0,50))

a = array([[1,4],[2,3]])
b = array([[4,1],[3,2]])
print a
print b
print multiply(a,b)

a = arange(16).reshape(4,4)
print mat(a).A
print mat(b).A
print nonzero(mat(a)[:,0].A == 0)[0]
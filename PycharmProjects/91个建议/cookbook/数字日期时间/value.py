#coding:utf8
import math
from fractions import Fraction
#保留小数点后几位
print round(1.23,1)
print round(1.27,1)
print round(-1.27,1)
print round(1.234556,4)
print format(1.23456,'0.2f')
#精确的小数计算
a = 4.2
b = 2.1
print (a + b) == 6.3
from decimal import Decimal
a = Decimal(4.2)
b = Decimal(2.1)
print a + b
print (a + b) == Decimal(6.3)

#二进制 八进制 十六进制
x = 1234

print bin(x)
print oct(x)
print hex(x)
print format(x,'b')
print format(x,'o')
print format(x,'x')

#复数的计算 real imag
a = complex(2,4)
b = 3 - 5j
print b.real
print b.imag
print a + b
print a/b
print a*b
print abs(a)
#无穷大数的计算
a = float("inf")
print a + 45
b = float("-inf")
print b + 12
print a + b
c = a + b
print math.isnan(c)

#分数问题
a = Fraction(5,4)
b = Fraction(7,16)
print a,b
print a*b
print a+b

c = a*b
print c.numerator
print c.denominator
print float(c)

#处理大型数组的计算
x = [1,2,3,4]
y = [5,6,7,8]
print x * 2
print x + y
import numpy as np

ax = np.array([1,2,3,4])
ay = np.array([5,6,7,8])
print ax * 2
print ax + 110
print ax + ay

#矩阵和线性代数的计算
m = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print m
print m.T
print m.I #矩阵的逆
v = np.matrix([[2],[3],[4]])
print v
print m*v
import numpy.linalg
print numpy.linalg.det(m)

#random.choice随机选择

import random
values = [1,2,3,4,5,6]
print random.choice(values)
print random.choice(values)
print random.choice(values)
print random.choice(values)
print random.sample(values,2)
print random.sample(values,2)
print random.sample(values,2)
print random.sample(values,3)
random.shuffle(values)
print values
random.shuffle(values)
print values
print random.randint(0,10)
print random.randint(0,10)
print random.randint(0,10)
print random.randint(0,10)

#时间换算

from datetime import timedelta
from datetime import datetime
a = timedelta(days=2,hours=6)
b = timedelta(hours = 4.5)
c = a + b
print c.days
print c.seconds
print c

d = datetime(2016,11,04)

print (d + timedelta(days=10))

b = datetime(2016,12,21)
a =b - d
print a.days
print datetime.today()
#将字符串转化成日期

text = '2016-11-04'
y = datetime.strptime(text,'%Y-%m-%d')
x = datetime.now()

diff = x - y

print diff


#coding:utf8
'''class Pair():
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __str__(self):
        return '({x!s},{y!s})'.format(x = self.x,y = self.y)

    def __repr__(self):
        return 'pair({x!r},{y!r})'.format(x = self.x,y = self.y)

p = Pair(2,3)
print p
#自定义字符串输出
_format = {
    'ymd':'{d.year}-{d.month}-{d.day}',
    'mdy':'{d.month}-{d.day}-{d.year}',
    'dmy':'{d.day}-{d.month}-{d.year}'
}
class Date():
    def __init__(self,year,month,day):
        self.year = year
        self.month = month
        self.day = day

    def __format__(self,code):
        if code == '':
            code = 'ymd'
        fmt = _format[code]
        return fmt.format(d = self)

d = Date(2012,12,21)
print format(d)

print format(d,'mdy')

print format(d,'dmy')
'''
#让对象支持上下文管理协议，上下文协议由with管理

#创建可管理的属性
'''class  Person():
    def __init__(self,first_name):
        self.first_name = first_name

    @property
    def first_name(self):
        return self.first_name

    @first_name.setter
    def first_name(self,value):
        if not isinstance(value,str):
            raise TypeError('Exception a string')
        self._first_name = value

    @first_name.deleter
    def first_name(self):
        raise AttributeError('Can not delete attribute')


a = Person('Guido')
print a.first_name

a.first_name = 42

del a.first_name

'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# blog.ithomer.net
'''
class Cls(object):
    def __init__(self):
        self.__x = None

    def getx(self):
        return self.__x

    def setx(self, value):
        self.__x = value

    def delx(self):
        del self.__x

    x = property(getx, setx, delx, 'set x property')


if __name__ == '__main__':
    c = Cls()
    c.x = 100
    y = c.x
    print("set & get y: %d" % y)

    del c.x
    print("del c.x & y: %d" % y)'''


#调用父类的方法
'''class a():
    def spam(self):
        print 'this is A'
    
class b(a):
    def spam(self):
        print 'This is B'
        super(b, self).spam()'''

'''class base():
    def __init__(self):
        print 'Base.init'

class a(base):
    def __init__(self):
        super().__init__()
        print 'a.init'
class b(base):
    def __init__(self):
        super().__init__()
        print 'b.init'

class c(a,b):
    def __init__(self):
        super().__init__()
        print 'c.init'
'''

#创建一种新形式的类属性或属性实例（类型检查属性）
'''class Integer():
    def __init__(self,name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else :
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value,int):
            raise TypeError('this is not int')
        instance.__dict__[self.name] = value

    def __del__(self,instance):
        del instance.__dict__[self.name]

class Point:
    x = Integer('x')
    y = Integer('y')

    def __init__(self,x,y):
        self.y = y
        self.x = x

p = Point(2,3)
print p.x
print p.y
p.y = 2.5
'''

#让属性具备惰性求值的能力，只有访问是才参与计算，访问了属性，把计算只缓存起来
class lazy():
    def __init__(self,func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance,self.func.__name__,value)
            return value



import math
class Circle():
    def __init__(self,radius):
        self.radius = radius

    @lazy
    def area(self):
        print ('compute area')
        return math.pi*self.radius**2

    @lazy
    def perimeter(self):
        print ('compute perimeter')
        return 2*math.pi*self.radius

c = Circle(2.0)
print (c.radius)

print (c.area)
print (c.area)

print (c.perimeter)
print (c.perimeter)

#简化数据结构的过程


#定义一个接口和抽象基类

from abc import ABCMeta,abstractclassmethod

class Isteam(metaclass=ABCMeta):

    @abstractclassmethod
    def first(self):
        pass

    @abstractclassmethod
    def second(self):
        pass


class sockestea(Isteam):

    def first(self):
        print('this is chongxie first')

    def second(self):
        print('this is chongxie second')

s = sockestea()
print(s.first())
print(s.second())


#实现自定义类，通过直接继承相应的内部类
import collections
import bisect

class SortefItem(collections.Sequence):
    def __init__(self,initial = None):
        self._item = sorted(initial) if initial else []

    def __getitem__(self, item):
        return self._item[item]

    def __len__(self):
        return len(self._item)

    def add(self,item):
        bisect.insort(self._item,item)


s = SortefItem([4,2,6,9,5])
print (list(s))

print (s[2])

print (len(s))
s.add(10)
print (list(s))

for i in s:
    print (i)


#委托属性的访问，作为继承的代替方案或者是实现一种代理机制

class A():
    def spam(self):
        print('this is A spam')

    def foo(self):
        print('this is A foo')

class B():
    def __init__(self):
        self._a = A()

    def spam(self):
        return self._a.spam()

    def foo(self):
        return self._a.foo()

    def bar(self):
        pass

b = B()
print(b.foo())
print(b.spam())

class A():
    def spam(self,x):
        print('this is A spam')

    def foo(self):
        print('this is A foo')


class C():
    def __init__(self):
        self._a = A()

    def bar(self):
        print('This is C bar')

    def spam(self,x):   #1
        print ('this is C spam')
        self._a.spam(x)

    def __getattr__(self, item): #2
        return getattr(self._a,item)

c = C()
print (c.bar())
print(c.spam(40))

#不用调用init创建实例
class Date:
    def __init__(self,year,month,day):
        self.year = year
        self.month = month
        self.day = day

d = Date.__new__(Date)
print(d)

s = {'year':2016,'month':11,'day':12}

for key,value in s.items():
    setattr(d , key ,value)

print (d.year,d.month,d.day)


#用mixin技术来扩展类定义
class LoggedMapping:
    __slots__=  ()
    def __getitem__(self, item):
        print('Getting'+str(item))
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        print('Setting {} = {!r}'.format(key,value))
        return super().__setitem__(key,value)
    def __delitem__(self, key):
        print('Deleting' + str(key))
        return super().__delitem__(key)

class SetOnceMapping:
    __slots__ = ()
    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(str(key) + 'already set')
        return super().__setitem__(key,value)

class StringkeyMapping:
    __slots__ = ()
    def __setitem__(self, key, value):
        if not isinstance(key,str):
            raise TypeError('key must be strings')
        return super().__setitem__()

class LoggedDict(LoggedMapping,dict):
    pass

d = LoggedDict()
d['x'] = 3
print(d['x'])
del d['x']

from collections import defaultdict
class SetOnceDefaultDict(SetOnceMapping,defaultdict):
    pass

d = SetOnceDefaultDict(list)
d['x'].append(3)
d['y'].append(2)
d['z'].append(12)
d['x'].append(10)


#实现带有状态的对象或状态机
class Connect():
    def __init__(self):
        self.new_state(CloseConnect)

    def new_state(self,cls):
        self._state = cls

    def read(self):
        return self._state.read(self)

    def write(self,data):
        return self._state.write(self,data)

    def open(self):
        return self._state.open(self)

    def close(self):
        return self._state.close(self)


class ConnectState():
    @staticmethod
    def read(self):
        raise NotImplementedError()

    @staticmethod
    def write(self):
        raise NotImplementedError()

    @staticmethod
    def open(self):
        raise NotImplementedError()

    @staticmethod
    def close(self):
        raise NotImplementedError()


class OpenConnect(ConnectState):
        @staticmethod
        def read(self):
            raise RuntimeError('reading')

        @staticmethod
        def write(self,data):
            raise RuntimeError('writting')

        @staticmethod
        def open(self):
            raise RuntimeError('already open')

        @staticmethod
        def close(self):
            self.new_state(CloseConnect)



class CloseConnect(ConnectState):
        @staticmethod
        def read(self):
            raise RuntimeError('not open')

        @staticmethod
        def write(self,data):
            raise RuntimeError('not open')

        @staticmethod
        def open(self):
            self.new_state(OpenConnect)

        @staticmethod
        def close(self):
            raise RuntimeError('already close')

c = Connect()
c.read()
c.open()
c.read()  #python __class__ 内部方法作用





#在环装数据结构中管理内存


#让类支持比较操作


#创建缓存实例




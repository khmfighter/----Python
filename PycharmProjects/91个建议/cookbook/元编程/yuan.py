#coding:utf8

#给函数加上一个包装,加入@wraps包可以保存元数据

import time
from functools import wraps

def timethis(func):

    @wraps(func)
    def wrapper(*args,**kagrs):
        start = time.time()
        result = func(*args,**kagrs)
        end = time.time()
        print(func.__name__,end - start)
        return result

    return wrapper


@timethis
def countdow(n):
    while(n>1):
        n -= 1

countdow(100000)

#对装饰器进行解包装
@timethis
def add(x,y):
    return x+y

add = add.__wrapped__
print (add(3,4))

#定义一个可接受收参数的装饰器

from functools import wraps,partial
import logging




def logged(level,name = None , message = None):
    def decorator(func):

        logname = name if name else func.__name__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args,**kwargs):
            log.log(level , logmsg)
            return func(*args,**kwargs)
        return wrapper
    return decorator


@logged(logging.DEBUG)
def add(x,y):
    return x+y

@logged(logging.CRITICAL)
def spam():
    print('spam')

print (add(1,2))



#定义一个可由用户修改的装饰器

def attach_wrapper(obj,func = None):
    if func is None:
        return partial(attach_wrapper,obj)
    setattr(obj,func.__name__,func)
    return func

def logged(level, name=None, message=None):
    def decorator(func):
        logname = name if name else func.__name__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)

        @attach_wrapper(wrapper)
        def set_level(newlevel):
            nonlocal level
            level = newlevel

        @attach_wrapper(wrapper)
        def set_message(newmsg):
            nonlocal logmsg
            logmsg = newmsg
        return wrapper

    return decorator


@logged(logging.DEBUG)
def add(x, y):
    return x + y


@logged(logging.CRITICAL,'example')
def spam():
    print('spam')


print(add(1, 2))


#在类中定义装饰器

class A:

    def decoratorA(seel,func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            print('decoratorA')
            return func(*args,**kwargs)
        return wrapper

    @classmethod
    def decoratorB(self,func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            print('deracotorB')
            return func(*args,**kwargs)
        return wrapper


a = A()

@a.decoratorA
def add(x,y):
    return x+y

@A.decoratorB
def sub(x,y):
    return x-y

print (add(1,2))
print (sub(2,1))

#把装饰器定义成类
import types
from functools import wraps

class Profiled:
    def __init__(self,func):
        wraps(func)(self)
        self.ncalls = 0

    def __call__(self, *args, **kwargs):
        self.ncalls += 1
        return self.__wrapped__(*args,**kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return types.MethodType(self ,instance)

@Profiled
def add(x,y):
    return x+y

class Spam:
    @Profiled
    def bar(self,x):
        print (self,x)

print(add(1,2))
s = Spam()
s.bar(1)
s.bar(2)


#把装饰器作用到类和静态方法上

import time
from functools import wraps


def timethis(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.time()
        r = func(*args,**kwargs)
        end = time.time()
        print (end - start)
        return r
    return wrapper

class Spam:
    @timethis
    def instance_method(self,n):
        print(self,n)
        while n>0:
            n -= 1

    @classmethod
    @timethis
    def class_method(cls,n):
        print(cls,n)
        while n>0:
            n -= 1


    @staticmethod
    @timethis
    def static_method(n):
        print (n)
        while n>0:
            n -= 1
s = Spam()

s.instance_method(10000)
s.instance_method(10000)
Spam.class_method(100)
Spam.static_method(1000)

#编写装饰器为被包装的函数添加参数
from functools import wraps

def opional_debug(func):
    @wraps(func)
    def wrapper(*args,debug = False,**kwargs):
        if debug:
            print('Calling',func.__name__)
        return func(*args,**kwargs)
    return wrapper

@opional_debug
def Spam(a,b,c):
    print(a,b,c)

print(Spam(1,2,3))
print(Spam(1,2,3,debug = True))




#使用装饰器给类定义打补丁
def log_getattribute(cls):

    #获取对象进行加工
    orig_getattribute = cls.__getattribute__

    def new_getattribute(self,name):
        print(self,name)
        return orig_getattribute(self,name)

    #加工完后将对象返回
    cls.__getattribute__ = new_getattribute

    return cls


@log_getattribute
class A:
    def __init__(self,x):
        self.x = x
    def spam(self):
        pass

a = A(12)
print (a.x)
a.x = 'x'
print(a.x)
print(a.spam())
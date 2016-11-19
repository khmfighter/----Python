#coding:utf-8
def convert(func, seq):
    print 'conv sequence of numbers to same type'
    return [func(eachNum) for eachNum in seq]
#解析式

myseq = (123,45.67,-6.2e8,99999999L)

print convert(int,myseq)
print convert(float,myseq)
print convert(long,myseq)

def MyFunc(name):
    print name + ' world'

MyFunc('hello')
list = MyFunc.__doc__
print list
print help(MyFunc)
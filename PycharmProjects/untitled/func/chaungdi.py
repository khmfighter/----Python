#coding:utf-8
#赋值
def foo():
    print 'this is foo'
bar = foo
bar()
print id(bar),id(foo)


#调用

def foo1():
    print 'this is foo1'

def bar1(x):
    x()

bar1(foo1)

print id(foo1),id(bar1)
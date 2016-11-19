#coding:utf8
'''from enum import Enum

Animal = Enum('Animal', 'ant bee cat dog')


class Animal(Enum):
    ant = 1
    bee = 2
    cat = 3
    dag = 4
 python 3.0以上版本可以使用此类枚举类
'''
def enum0(**first):
    return type('enum0',(),first)

Number0 = enum0(One = 1,Two = 2,Three = 3)
print Number0.One
print Number0.Two
print Number0.Three


def enum1(*first , **second):
    dict_test = dict(zip(first,range(len(first))),**second)
    return type('enum1',(),dict_test)


Number1 = enum1('One','Two','Three')

print Number1.One
print Number1.Two
print Number1.Three

def enum2(*first,**second):
    enums = dict(zip(first,range(len(first))),**second)
    reversed = dict((value,key) for key,value in enums.iteritems())
    enums['revered'] = reversed
    return type('enum2',(),enums)

Number2 = enum2('One','Two','Three')
print Number2.revered[1]
print Number2.revered[2]
print Number2.revered.iteritems()
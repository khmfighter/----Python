#coding:utf8
class UpperString(object):

    def __init__(self):
        self.value = ''
    def __set__(self, instance, value):  #把值付给对象
       self.value = value.upper()
    def __get__(self, instance, owner): #从对象取值
        return self.value
class MyDescription():
    pass
class MyClass(object):
    attribute = UpperString()

instance_of = MyClass()
print instance_of.attribute


instance_of.attribute = 'hello world'
print instance_of.attribute

instance_of.__dict__ = {}

instance_of.new_att = 1
print instance_of.__dict__

MyClass.new_att = MyDescription()
print instance_of.new_att

print dir(UpperString)
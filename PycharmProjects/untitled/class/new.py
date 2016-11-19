class MyClass(object):
    def __new__(cls, *args, **kwargs):
        print '__new__ called'
        return object.__new__(cls)
    def __init__(self):
        print '__init__ called '
        self.a = 1

instance = MyClass()
print instance.a

class MyOtherClass(MyClass):
    def __init__(self):
        print 'MyOther class __init__ called'
        super(MyOtherClass,self).__init__()
        self.b = 2

instance1 = MyOtherClass()
print instance1.a
print instance1.b
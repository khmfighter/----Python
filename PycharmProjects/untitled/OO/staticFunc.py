class TestStaticMethod(object):
    def foo():
        print 'calling static method foo()'

    foo = staticmethod(foo)

class TestClassMethod():
    def foo(cls):
        print 'calling class method foo()'
        print 'foo() is part of class :',cls.__name__

    foo = classmethod(foo)

tsm = TestClassMethod()
TestClassMethod.foo()
tsm.foo()

tcm = TestClassMethod()
TestClassMethod.foo()
tcm.foo()
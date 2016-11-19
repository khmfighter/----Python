class P(object):
    def foo(self):
        print 'P-foo()'

class C(P):
    def foo(self):
        print 'C-foo()'

c = C()
c.foo()

p = P()
P.foo(c)

class D(P):
    def foo(self):
        super(D,self).foo()
        print 'D-foo()'


d = D()
d.foo()
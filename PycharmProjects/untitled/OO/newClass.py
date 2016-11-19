class MyClass (object):
    "hello world "
    version = 1.1
    def MyVersion(self):
        pass


c = MyClass()
print c.__class__.__name__
print c.__doc__
print c.__dict__
print c.__module__
print c.__class__.__base__


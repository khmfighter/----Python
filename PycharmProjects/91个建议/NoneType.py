class A():
    def __nonzero__(self):
        print 'no empty'
        return True
    def __len__(self):
        print 'get length'
        return False

if A():
    print 'not empty'
else:
    print 'empty'

print len('abc')
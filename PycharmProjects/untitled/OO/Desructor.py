class Destrutor(object):
    count = 0
    def __init__(self):
        Destrutor.count += 1

    def __del__(self):
        Destrutor.count -= 1

    def total(self):

        return Destrutor.count

a = Destrutor()
print a.total()
b = Destrutor()
c= Destrutor()
print c.total()

del c
print b.total()

print dir(Destrutor)
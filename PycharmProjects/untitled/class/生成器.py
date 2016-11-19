#coding:utf8
def myGen():
    print '生成器'
    yield  1
    yield  2

myG = myGen()
print next(myG)
print next(myG)


for i in myGen():
    print i


def libs():
    a = 0
    b = 1
    while True:
        a,b = b,a+b
        yield a

for each in libs():
    if each>100:
        break;
    print each
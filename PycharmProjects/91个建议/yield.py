def func():
    a,b = 0,1
    while a<=100:
        yield a
        a , b = b , a+b
for w in func():
    print w

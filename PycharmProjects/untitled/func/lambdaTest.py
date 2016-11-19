def ds(x):
    return 2*x+1
print ds(5)

g = lambda x : x*2+1

print g(5)


g = lambda x,y : x+y
print g(3,4)

t = [1,2,3,4,5,6]
g = filter(lambda x :x>3,t)
print g
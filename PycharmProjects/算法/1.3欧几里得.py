#coding:utf8
m = raw_input("imput m:")
n = raw_input("imput n:")

r = int(m) % int(n)


while (r != 0):
    m = n
    n = r
    r = int(m)%int(n)

print "最小公约数为",n
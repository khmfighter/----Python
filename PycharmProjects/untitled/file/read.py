f = open("imooc.txt")

#print f.read()

#print f.read(10)

#print f.readline(100)

#print f.readline(2)
#   print each
iter_f = iter(f)
lines = 0


for line in iter_f:
    lines += 1
    print lines
f.close()
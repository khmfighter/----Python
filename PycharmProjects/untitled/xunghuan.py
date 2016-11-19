

for i in range(0,100):
    print("Item {0},{1}.".format(i,"HelloWorld"))

album = ('hello','world','haha','legendary')
for i  in sorted(album):
    print i

for i in reversed(album):
    print i

for i,albums in enumerate(album):
    print i,albums

print [ (x+1,y+1) for x in range(3) for y in range(6)]
seq = [11,10,8,9,6,7,9,11]
print filter(lambda x: x%2 ,seq)
print [x for x in range(11) if x%2]
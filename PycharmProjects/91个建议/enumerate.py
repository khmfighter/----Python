li = ['a','b','c','d','e']

'''
for i,e in enumerate(li):
    print 'index',i,'element',e

enum = enumerate(li)
print enum.next()
print enum.next()
'''

def enumerate1(sequence,start=0):
    n =start
    for num in sequence[n:]:
        yield n,num
        n += 1

num = enumerate1(li,2)
print num.next()
print num.next()



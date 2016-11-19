dict1 = {}
dict2 = {'hello':'world','java':'pathon'}
print dict1,':',dict2
print dict2.has_key('hello')
dict1.setdefault('nima','hehe')
print dict1
dict2.setdefault('1','1.0')
print dict2
fdict = dict((['x', 1], ['y', 2]))
print fdict

ddict = {}.fromkeys(('x','y','z'),-1)
print ddict
edict = {}.fromkeys(('foo','bar'))
print edict

for i in dict2.keys():
    print 'keys:',i,'values:',dict2[i]

print cmp(dict1,dict2)

a = dict(zip(('x','y'),(1,2)))
print a

dict1.update(dict2)
print dict1


s = set('hello world')
print s
s.remove('h')
print  s
s -= set('o')
print s

print dir(set)
print dir(dict)
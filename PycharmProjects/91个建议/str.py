assert 'Hello World'.istitle() == True
assert  'Hello world'.istitle() == False

str = 'test if a string contains some special substring'

if str.find("some") != -1:
    print 'yes, contains it'
if "some" in str :
    print "some have in str"

str1 = 'hello world'
print str1.split()
print str1.title()


a = ['1',1,'a',3,7,'n']

print sorted(a)
print a

print a.sort()
print a

person = [{'name':'Jon','age':23},{'name':'Alan','age':32},{'name':'Bob','age':12}]

print sorted(person,key = lambda x :(x['age'],x['name']))

L = [('b',1),('a',2),('c',4),('d',3)]

print sorted(L,cmp = lambda x,y: cmp(x[0],y[0]))
print sorted(L,key=lambda x:x[0])
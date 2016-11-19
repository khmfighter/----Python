s = '{0} love {1}'.format('i','you')
print s
s = '{a} love {b}'.format(a='i',b='you')
print s

print '\ta'
print '{{0}}'.format('budayin')
#print '{.2f}{1}'.format(27.658,'GB')


print '%c' % 97
print '%c %c %c' % (97,98,99)

print '%s' % 'hehe'

tuple = (1,2,3,4,5,56,76,8,89,9)
print sum(tuple)

tuple = (1,2,3,4,5,56,76,8,89,9)
print sum(tuple,100)

tuple = (1,2,3,4,5,56,76,8,89,9)
print sorted(tuple)

tuple = (1,2,3,4,5,56,76,8,89,9)
print list(reversed(tuple))
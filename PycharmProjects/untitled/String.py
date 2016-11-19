import string
a = 'abcdefg'
print a[::-1]
print a[::2]

s = 'abcde'
for i in range(-1,-len(s),-1):
    print s[:i]

s = 'abcde'
for i in [None]+range(-1,-len(s),-1):
    print s[:i]


print string.ascii_letters
print string.ascii_lowercase
print string.ascii_uppercase
print string.digits
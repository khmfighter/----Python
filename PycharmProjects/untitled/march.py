import re
secret_code='fasdffwxxixxegvsfgsfaxxlovexxsdfawsfwxxyouxxgasgasdfasdfasdfasdfasdfasdfa'
a='xyxy123'
b=re.findall('x*',a)
print b

a='xy123'
b=re.findall('x.',a)
print b

a='xy123'
b=re.findall('x?',a)
print b

b=re.findall('xx.*xx',secret_code)
print b

b=re.findall('xx.*?xx',secret_code)
print b

b=re.findall('xx(.*?)xx',secret_code)
print b
for each in b:
    print each

s='''sadfsadfaxxhello
xxasdfasxxworldxxdsafs'''
d=re.findall('xx(.*?)xx',s,re.S)
print d

s2='sfasdfasfxxIxx123xxlovexxfsdaf'
f=re.search('xx(.*?)xx123xx(.*?)xx',s2).group(1)
print f
s2='sfasdfasfxxIxx123xxlovexxfsdaf'
f2=re.findall('xx(.*?)xx123xx(.*?)xx',s2)
print f2[0][1]
import re

s = re.match('foo','foo')
if s is not None:
    print s.group()

s = re.match('foo','bar')
if s is not None:
    print s.group()

s = re.match('foo','hello foo  ok')
if s is not None:
    print s.group()

s = re.match('foo', 'foo hello ok')
if s is not None:
    print s.group()

patern = '\w+\@(\w+\.)?\w+\.com'
s = re.match(patern,'baidu@www.xxx.com')
if s is not None:
    print s.group()

patern = '\w+\@(\w+\.)*\w+\.com'
s = re.match(patern,'baidu@www.xxx.asbc.com')
if s is not None:
    print s.group()

s = re.match('(a(b))','ab')

s = re.search(r'\bthe','hello the asfsadf')
if s is not None:
    print s.group()

s = re.search(r'\Bthe','the asfsadf')
if s is not None:
    print s.group()

s = re.findall('car', 'carry the barcardi to the car')
print s
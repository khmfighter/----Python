import re

m = re.search(r'\bhello','hello world')
if m is not None :
    print m.group()


m = re.search(r'\Borld','hello world')
if m is not None :
    print m.group()

m = re.search('\Ahello','hello world')
if m is not None :
    print m.group()

net = '\w+@\w+\.com'
m = re.match(net,'nobody@xxx.com')
if m is not None:
    print m.group()

import re

s  = re.search('foo','helle fooname')
if s is not None:
    print s.group()
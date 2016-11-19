import re

s = re.findall('(?i)yes','yes? Yes. YES!!')
print s

s = re.findall('(?i)th[\w]+','The question way is through this tunnel')
print s

print re.findall(r'(^th[\w ]+)',"""
    This line is the first
    another line
    that line is final,it is the best
""",re.M)

s = re.findall(r'(?i)th.+','''
The first line
the second line
the third line
''')
print s

s = re.findall(r'(?s)th.+','''
The first line
the second line
the third line
''')
print s
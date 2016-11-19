#coding:utf8
filename = 'file.txt'
print filename.endswith('.txt')
print filename.startswith('file:')

url = 'http://www.python.org'
print url.startswith('http:')


choice = ['http:','ftp:']
url = 'http://www.python.org'
print url.startswith(tuple(choice))

from fnmatch import fnmatch,fnmatchcase
print fnmatch('hello.txt','*.txt')
print fnmatch('foo.txt','?oo.txt')
print fnmatch('Dat45.csv','Dat[0-9][0-9]*')

#预编译的正则表达式匹配
import re
detach = re.compile(r'(\d+)/(\d+)/(\d+)')
m = detach.match('11/12/2013')
print m.group(0)
print m.group(1)
print m.group(2)
print m.group(3)
print m.groups()

#re.repalce替换
text = 'yeah,but no but yeah,but,no'
text.replace('no','hello')
print text


text = 'hello PYTHON HELLO python yes Python'

Ignore = re.findall('python',text,re.IGNORECASE)
print Ignore


#对齐文本字符串
text = 'hello world '

print text.ljust(20)
print text.rjust(20)
print text.center(20)
print text.rjust(20,'=')
print text.center(20,'-')
print format(text,'>20')
print format(text,'<20')
print format(text,'=>20')



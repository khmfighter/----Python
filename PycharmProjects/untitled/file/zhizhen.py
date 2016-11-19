f = open('2.txt','r+')
import os

print f.tell()

print f.read(3)

print f.tell()

f.seek(0,os.SEEK_SET)

print f.tell()

f.seek(0,os.SEEK_END)

print f.tell()
print f.read()
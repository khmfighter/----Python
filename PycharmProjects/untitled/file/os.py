#coding:utf-8
import os
fd = os.open('im.txt',os.O_CREAT|os.O_RDWR)
n = os.write(fd,'imooc')
i = os.lseek(fd,0,os.SEEK_SET)
str1 = os.read(fd,5)
print str1

#判断是否有权限
print os.access('im.txt',os.F_OK)
print os.access('im.txt',os.R_OK)
print os.access('im.txt',os.W_OK)
print os.access('im.txt',os.X_OK)

print os.listdir('/')

os.mkdir('test1')
os.makedirs('test1/test1/test2')
os.removeddirs('test1/test1/test2')

os.exists('./imooc.txt')

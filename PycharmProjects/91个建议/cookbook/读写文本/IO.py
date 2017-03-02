#coding:utf8
with open('hello.txt','wt') as f:
   print ('hello world',file=f)

#以不同的分隔符或行结尾符完成打印
print('acme',50,98.5)
print('acme',50,98.5,sep=',')
print('acme',50,98.5,sep=',',end='!!!/n')
s1 = ['acme',50,98.5]
s = ','.join(str(x) for x in s1)
   print (s)

   # 序列化pickle
   import os
   import pickle

   f = open('simple', 'wb')
   pickle.dump('hello', f)
   s = open('simple', 'rb')
   str1 = pickle.loads(s)
   print(str(str1))



#coding:utf8
try:
   with open('2.txt','r') as f: #with可以自动关闭资源，等于finally模块的作用
    for each_line in f.readlines():
        print each_line
except OSError as reason:
    print reason
#finally:
 #   f.close()
#coding:utf8
m = raw_input("please input a number:")
n = raw_input("please input other number:")
t = int(min(m,n))
while(1):
    if(int(m)%t == 0):
        if(int(n)%t == 0):
            print t,"是最大公约数"
            break
        else:
            t = t-1
    else:
        t =t-1
#coding=utf-8
import threading
from time import sleep,ctime


class MyThread(threading.Thread):
    def __init__(self,func,args,name=""):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
    def getResult(self):  #获取结果在执行
        return self.res

    def run(self):
        print 'starting ',self.name,'at :',ctime()
        self.res = self.func(*self.args)    #执行对应线程绑定的函数，并return结果回来
        print self.name,'finished atg:',ctime()
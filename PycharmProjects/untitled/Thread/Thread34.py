#coding=utf-8
import threading
from time import sleep,ctime

class MyFunction(threading.Thread):
    def __init__(self,func,args,name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
    def run(self): #重写父类run方法，在线程启动后执行该方法内的代码
        self.func(*self.args)

loops = [4,2]

def loop(nloop,nsec):
    print 'start loop',nloop,'at:',ctime()
    sleep(nsec)
    print 'loop',nloop,'done at:',ctime()


def main():
    print 'starting at:',ctime()
    threads=[]
    nloops = range(len(loops))
    for i in nloops:
        t = MyFunction(loop,(i,loops[i]),loop.__name__) #关键点
        threads.append(t)
    for i in nloops:
        threads[i].start()
    for i in nloops:
        threads[i].join()
    print 'all DONE at:',ctime()

if __name__ == '__main__':
    main()
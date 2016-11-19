#coding=utf-8
import threading
from time import sleep,ctime

class ThreadFunc(object):
    def __init__(self,func,args,name=''):
        self.name = name
        self.func = func
        self.args = args
    def __call__(self):  #调用自身
        self.func(*self.args)  #多个参数

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
        t = threading.Thread(target=ThreadFunc(loop,(i,loops[i]),loop.__name__)) #关键点
        threads.append(t)
    for i in nloops:
        threads[i].start()
    for i in nloops:
        threads[i].join()
    print 'all DONE at:',ctime()

if __name__ == '__main__':
    main()
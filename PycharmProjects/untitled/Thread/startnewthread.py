from time import sleep,ctime
from thread import start_new_thread
def loop0():
    print 'start loop 0 at:w',ctime()
    sleep(4)
    print 'loop 0 done at:',ctime()

def loop1():
    print 'start loop 1 at:w',ctime()
    sleep(2)
    print 'loop 1 done at:',ctime()

def main():
    print 'starting at:',ctime()
    start_new_thread(loop0,())
    start_new_thread(loop1,())
    sleep(6)
    print 'all done at:',ctime()
if __name__ == '__main__':
    main()
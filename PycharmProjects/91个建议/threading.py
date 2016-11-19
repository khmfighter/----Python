import sys
import threading
import time

class test(threading.Thread):
    def __init__(self,name,delay):
        threading.Thread.__init__(self)
        self.name = name
        self.delay = delay

    def run(self):
        print "%s delay for %s"  % (self.name,self.delay)
        time.sleep(self.delay)
        c = 0
        while True:
            print "This is thread %s on line %s" %(self.name,c)
            c = c + 1
            if c==3:
                print "End of thread %s" % self.name
                break
import traceback
import sys
import logging

'''logging.basicConfig(
    level = logging.DEBUG,
    filename = 'log.txt',
    filemod = 'w',
    format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
)'''
gList = ['a','b','c','d','e','f','g']
def f():
    gList[5]
    logging.info('[INFO] : calling method g() in f()')
    return g()
def g():
    return h()
def h():
    del gList[2]
    return i()
def i():
    gList.append('i')
    print gList[7]

if __name__ == '__main__':
    try :
        f()
    except IndexError as ex:
        print "this is a error"
        print ex
        traceback.print_exc()
import traceback
gList = ['a','b','c','d','e','f','g']
def f():
    gList[5]
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
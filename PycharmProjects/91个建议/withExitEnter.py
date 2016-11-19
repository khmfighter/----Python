#coding:utf8
class MyContextmManager(object):
    def __enter__(self):
        print 'enter...'
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print 'no problem'
            return False      #异常被重新抛出
        elif exc_type is ValueError:
            print 'This is a ValueError'
            return True      #异常被挂起，程序继续执行
        else:
            print 'other error'
            return True

'''with MyContextmManager():
    print 'testing ....'
    raise(ValueError)
'''
with MyContextmManager():
    print 'testing '
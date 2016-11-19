#coding:utf8
#组合
class Turtle(object):
    def __init__(self,x):
        self.num = x
class Fish(object):
    def __init__(self,y):
        self.num = y
class Pool(object):
    def __init__(self,x,y):
        self.turtle = Turtle(x)
        self.fish = Fish(y)
    def  print_num(self):
        print '水池里的乌龟有%d只 有鱼%d只' % (self.turtle.num,self.fish.num)

pool = Pool(10,20)

pool.print_num()
passline = 60
def func(val):
    passline = 90
    if val >= passline:
        print 'pass'
    else:
        print 'failed'
    def fun():
        print val
    fun()

def Max(val1,val2):
    return max(val1,val2)

func(89)
print Max(90,100)
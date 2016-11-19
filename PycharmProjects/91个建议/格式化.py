import math
print 'your score is %06.1f' % 9.5

itemname = 'circumference'
radius = 3
print 'the %s of a circle with radius %f is %0.3f' % (itemname,radius,math.pi*radius*2)

itemdict = {'circumference':'circumference','radius':3,'value':math.pi*radius*2}
print 'the %(circumference)s of a circle with radius %(radius)f is %(value)0.3f' % (itemdict)


print 'the {itemname} of a circle with radius {radius} is {value}'.format(itemname = 'circumference',radius = 3,value = math.pi*radius*2)

class Customer(object):
    def __init__(self,name,gender,phone):
        self.name = name
        self.gender = gender
        self.phone = phone

    def __str__(self):
        return 'Customer({self.name},{self.gender},{self.phone})'.format(self = self)

print str(Customer('ouomingwang','man','123456'))
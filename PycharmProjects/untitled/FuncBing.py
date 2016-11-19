from types import MethodType
class Student(object):

    def Student(self,name):
        self.name = name
        return name

def set_age(self,age):
        self.age = age
        return age
s1 = Student()
s1.name = 'hehe'
print s1.name

s1.set_age = MethodType(set_age,s1)
str = s1.set_age('haha')
print str

Student.student = Student

s2 = Student()

print s2.Student('enen')
s3 = Student()
print s3.Student('GG')



class Teacher(object):
    __slots__ = ('name','age')

t1 = Teacher()
t1.age = 100
t1.name = 1100
t1.score = 100
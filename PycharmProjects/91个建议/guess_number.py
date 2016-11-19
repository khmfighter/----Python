import random

name = raw_input("please input a name :")

number = random.randint(1,20)

print '{name1} 1 to 20'.format(name1 = name)

while True:
     guess_num = int(raw_input('please input a number:'))

     if number > guess_num:
         print 'too low'
     if number < guess_num:
         print 'too high'
     if number == guess_num:
         break

if guess_num == number:
    print "right!!"


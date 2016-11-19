import random

num = random.uniform(1,100)

while True:
    guess = int(raw_input("please input a num: "))
    if num == guess:
        print "you are right!!!"
        break
    if num>guess:
        print "smaller"
    if num<guess:
        print "bigger"
    print "\n"
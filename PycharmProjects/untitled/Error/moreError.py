try:
    f = open("2.txt")
    line = f.read(2)
    num = int(line)
    print "read num=%d"%num
except IOError,e:
    print "catch Error:",e
except ValueError,e:
    print "catch Error:",e
else:
    print "No Error"
finally:
    try:
        print "file is close"
        f.close()
    except NameError,e:
        print "catch Error:",e
f = open('1.txt','r+')
print type(f)

#print f.read(1)

string = raw_input("-> ")
if not string:
    f.write(string)

f.close()

print string
f = open('imooc1.txt','w')

f.write('text write')
f.writelines('123456')
f.writelines(('1','2','3','4','5','6'))
f.writelines(['1','2','3','4','5','6'])
f.writelines('123456')
f.close()

f1 = open('imooc1.txt','r')

str = iter(f1)

for i in str:
    print i
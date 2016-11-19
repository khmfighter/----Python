import pickle

lista=["mingyue","jishi","you"]
listb=pickle.dumps(lista)
print listb

listc=pickle.loads(listb)
print listc


group1=("bajiu","wen","qigntain")
f1=open('1.pk1','wb')
pickle.dump(group1,f1,True)
f1.close()

f2=open('1.pk1','rb')
t=pickle.load(f2)
print t
f2.close()
#coding:utf8
lista = [10,11,90,88,76,97,54,60,23,54]
print type(lista)
n = 10
i=0
while(i<n):

    index = i
    j = i+1
    while(j<n):
        if(lista[index]>lista[j]):
            index = j
        j += 1
    if(index != i):
        lista[index],lista[i] = lista[i],lista[index]

    i +=1
print lista


#coding:utf8
lista  = [19,22,10,60,54,89,76,12,65,23]

n = 10
i=0

while(i<n):
    j = 0
    while(j<n-i-1):
        if(lista[j+1]<lista[j]):
            lista[j+1],lista[j] = lista[j],lista[j+1]
        j += 1
    i += 1

print lista


#改进的冒泡排序
listb  = [19,22,10,60,54,89,76,12,65,23]
exchange = 10
while(exchange):

    bound = exchange
    exchange = 0
    k = 0

    while(k<bound-1):
        if(listb[k+1]<listb[k]):
            listb[k+1],listb[k] = listb[k],listb[k+1]
        exchange = k+1
        k += 1

print listb


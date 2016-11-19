def Mergesort(r,temp,s,t):
    if(s == t):
        temp[s] = r[s]
    else:
        mid = (s+t)/2
        Mergesort(r,temp,s,mid)
        Mergesort(r,temp,mid+1,t)
        print r
        print temp
        Merge(temp,r,s,mid,t)


def Merge(r,temp,s,mid,t):
    i = s
    j = mid + 1
    k = s
    while(i<=mid & j<=t):
        if(r[i]<=r[j]):
            temp[k] = r[i]
            i += 1
            k += 1
        else:
            temp[k] = r[j]
            j += 1
            k += 1

    if(i<=mid):
        while(i<=mid):
            temp[k] = r[i]
            k += 1
            i += 1
    else:
        while(j<=t):
            temp[k] = r[j]
            k += 1
            j += 1

r = [23,11,90,4,7,6,9,8,19]
temp = [0,0,0,0,0,0,0,0,0]
Mergesort(r,temp,0,8)
print r
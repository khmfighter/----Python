def nlargest(nlist,nums):
    list_nums= []
    tlist = list(nlist)
    for n in range(nums):
        number = tlist[0]
        for lista in tlist:
            if number < lista:
                number = lista
        list_nums.append(number)
        tlist.pop(tlist.index(number))
    return list_nums


def nsmallest(nlist,nums):
    list_nums= []
    tlist = list(nlist)
    for n in range(nums):
        number = tlist[0]
        for lista in tlist:
            if number < lista:
                number = lista
        list_nums.append(number)
        tlist.pop(tlist.index(number))
    return list_nums


lista = [1,2,4,5,6,7,8,9]
print nlargest(lista,3)
listb = [-1,-9,43,7,9,32,5,7,8,3,1]
print nsmallest(listb,8)
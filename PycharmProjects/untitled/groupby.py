from itertools import groupby

def compress(data):
    return ((len(list(group)),name) for group,name in groupby(data))

def decompress(data):
    return (size * car for car,size in data)


str = compress('get uuuuuuuuuuuuuuuup')
list1 = list(str)
print list1

compressed = compress('get uuuuuuuuuuuuuuuup')
print ''.join(decompress(compressed))
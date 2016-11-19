#coding:utf8
#简单匹配行，输出匹配


from collections import deque

'''
def search(lines,pattern,history = 5):
    pre_lines = deque(maxlen=history)
    for line in lines:
        if pattern in line:
            yield line,pre_lines
        pre_lines.append(line)

if __name__ == '__main__':
    with open('hello.txt') as f :
        for line , pre_lines  in search(f,'python',5):
            for pline in pre_lines:
                print pline
            print line
            print '_'*20
'''
#deque的用途
q = deque(maxlen=3)
q.append(1)
q.append(2)
q.append(3)

print q

q.append(4)
print q

q.appendleft(5)
print q

q.pop()
print q

q.popleft()
print q

#一个键值对应多个值（list or set）
from collections import defaultdict
d = defaultdict(list)
d['a'].append(1)
d['a'].append(2)
print d
d['b'].append(3)
print d

s = defaultdict(set)
s['a'].add(1)
s['a'].add(2)
s['a'].add(1)
s['b'].add(2)
print s


d = {}
d.setdefault('a',set()).add(1)
d.setdefault('a',set()).add(1)
d.setdefault('b',[]).append(2)
print d



#字典的排序 zip（）构建字典
dict_test = {
    'a':1,
    'b':2,
    'c':3,
    'd':4,
}

dicta = zip(dict_test.values(),dict_test.keys())
print dicta

print min(dicta),max(dicta)
print sorted(dicta,reverse=True)


#find keys in common
a = {
    'x':1,
    'y':2,
    'z':8
}
b = {
    'x':11,
    'y':2,
    'w':10
}

dicta = set(a.keys()) & set(b.keys())
print dicta
print  set(a.keys()) - set(b.keys())
print set(a.items()) & set(b.items())

#切片对象
slice_test = [0,1,2,3,4,5,6,7,8]
a = slice(2,4)
print slice_test[2:4]
print slice_test[a]
slice_test[a] = [10,11]

print  slice_test

del slice_test[a]
print slice_test

s ='helloworld'
print a.indices(len(s))

from collections import Counter
from operator import itemgetter

word = ['a', 'b', 'c', 'a', 'b', 'b']

word_counter = Counter(word)
print word_counter
#two_max = word_counter.most_common(1)
#print two_max

num_index = {}
for num in word:
    if num not in num_index:
        num_index[num] = 1
    else:
        num_index[num] += 1
dict_test = zip(num_index.values(),num_index.keys())

print sorted(dict_test,reverse=True)



#将名字映射到序列的元素中
from collections import namedtuple
Subscribe = namedtuple('Subscribe',['addr','joined'])
sub = Subscribe('oumingwang@163.com','2012-12-12')
print sub
print len(sub)
addr,joined = sub
print addr,joined

"""from collections import ChainMap

a = {'a':1,'z':2}
b = {'b':2,'z':3}

c = ChainMap(a,b)
#first a next b
字典合并
"""

a = {'a':1,'z':2}
b = {'b':2,'z':3}

b.update(a)
print b
phone = [1,2,3,5,5,6,7,8,9,6,5,3,6,8,9,7,4,4,6]
classList={}
for i in phone:
    if i in classList:
        classList[i] += 1
    else:
        classList[i] = 1

print classList
from collections import  defaultdict
count_frq = defaultdict(int)

for item in phone:
    count_frq[item] += 1

print count_frq
from collections import Counter
c = Counter('success')
print c

print list(c.elements())

print c.most_common(2)

c.update('setup sfawffasfasf')
print c
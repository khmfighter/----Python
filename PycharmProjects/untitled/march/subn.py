import re
s = re.sub('x','Mr.ou','attn: x\n\nDear x,\n')
print s

s = re.subn('x','Mr.ou','attn: x\n\nDear x,\n')
print s

s = re.sub('[a-e]','x','abcdef')
print s
s = re.sub('[ae]','x','abcdef')
print s

s = re.subn('[a-e]','x','abcdef')
print s
s = re.subn('[ae]','x','abcdef')
print s
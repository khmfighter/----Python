import re
s = re.split(':','str1:str2:str3')
print s

date = ('qionghai,Ca 10000',
        'haikou,ca',
        'sanya,10001',
        'lingshui 10002',
        'wenchang cb'
        )
for Date in date:
        print re.split(',|(?= (?:\d{5}|[a-z]{2}))|',Date)
import requests
import re
html = requests.get('http://tieba.baidu.com/f?ie=utf-8&kw=python')
print html.text

txt = re.findall('class="topic_name">(.*?)</a>',html.text,re.S)
for each in txt:
    print each

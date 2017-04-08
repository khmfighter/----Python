from pandas import DataFrame,Series
import pandas as pd

s = Series([1,2,3.0,'abc'])
print s

s = Series(data=[1,3,5,7],index=['a','b','c','d'])
print s
print s.index
print s.values

s.name = 'hello'
s.index.name = 'helloworld'
print s
s['helloworld'] = 2
print s

data = {'state':['Ohior','Ohior','Ohior','nevada','nevada',],
        'year':[2000,2001,2002,2001,2002],
        'pop':[1.5,1.7,3.6,2.4,2.9]
        }
df = DataFrame(data)
print df
'''
df = DataFrame(data,index = ['one','two','three','four','five'],
               columns=['year','state','pop','debt'])

print df
print df.index
print df.columns
print type(df['debt'])
'''

s = Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
a = ['a','b','c','d','e']
print s.reindex(a)
print s.reindex(a,fill_value=0.0)
#print s.reindex(a,method='ffill')

state = ['Texas','Utha','California']
print df.reindex(columns=state,method='ffill')
#print df.reindex(index=['a','b','c','d'],columns=state,method='ffill')

print df

a = ['a','b','c','d','e']
df = DataFrame(df,index=a)
print df
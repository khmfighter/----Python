import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


trainSet = pd.read_csv('train.csv')
print trainSet['SalePrice'].describe()
#print trainSet.columns
'''
print sns.distplot(trainSet['SalePrice'])
print("Skewness: %f" % trainSet['SalePrice'].skew())
print("Kurtosis: %f" % trainSet['SalePrice'].kurt())


var = 'GrLivArea'
data = pd.concat([trainSet['SalePrice'], trainSet[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#plt.show()

var = 'OverallQual'
data = pd.concat([trainSet['SalePrice'], trainSet[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
'''

'''
corrmat = trainSet.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
#plt.show()

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(trainSet[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(trainSet[cols], size = 2.5)
#plt.show()



total = trainSet.isnull().sum().sort_values(ascending=False)
percent = (trainSet.isnull().sum()/trainSet.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print missing_data.head(20)

df_train = trainSet.drop((missing_data[missing_data['Total'] > 0]).index,1)
#df_train = trainSet.drop(df_train.loc[df_train['Electrical'].isnull()].index)
print df_train.isnull().sum().max()
'''

sns.distplot(trainSet['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainSet['GrLivArea'], plot=plt)
plt.show()
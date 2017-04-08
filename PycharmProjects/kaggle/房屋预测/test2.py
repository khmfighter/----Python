import pandas as pd # data frames
import numpy as np # arrays and computing
import seaborn as sns # statistical data visualization
import matplotlib

import matplotlib.pyplot as plt # plots
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from scipy.stats import norm
from scipy import stats

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression, ElasticNetCV
from sklearn.model_selection import cross_val_score
#import xgboost as xgb



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print np.shape(train)
print train.columns
print train.head()
print train['SalePrice'].describe()
print train['YrSold'].describe()


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                        test.loc[:, 'MSSubClass':'SaleCondition']))

print np.shape(all_data)

all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = np.log(train.SalePrice)
print y


total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print missing_data.head(25)

train1 = train.copy()
all_data_t = all_data.copy()

all_data_t = all_data_t.drop((missing_data[missing_data['Total'] > 200]).index,1)
quantitative = [f for f in all_data_t.columns if all_data_t.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in all_data_t.columns if all_data_t.dtypes[f] == 'object']
all_vars = quantitative + qualitative
print all_vars[:10,:20]
#coding:utf8
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('train.csv')
print train_df.columns
print train_df.shape

test_df = pd.read_csv('test.csv')
print test_df.columns
print test_df.shape

'''
train_df = train_df.drop(['PassengerId','Name'],axis=1)
test_df = test_df.drop(['PassengerId','Name'],axis=1)
print train_df.shape
print test_df.shape
'''

combine = [train_df,test_df]
print np.shape(np.mat(combine))

#print train_df.head()
#print train_df.info()
#print test_df.info()
#print train_df.describe()
#print train_df.describe(include=["0"])
#选出数值型数据
#print train_df.describe(include=['O'])

#print train_df.describe(include=[object])

print train_df[['Pclass','Survived']].groupby(['Pclass'],as_index = False).mean().sort_values(by='Survived',ascending=False)

#print train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(['Survived'],ascending=False)

#print train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(['Survived'],ascending=False)

#print train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(['Survived'],ascending=False)

#g = sns.FacetGrid(train_df,col='Survived')
#g.map(plt.hist,'Age',bins=30)


#grid = sns.FacetGrid(train_df,col='Survived',row = 'Pclass',size=2.2,aspect=1.6)
#grid.map(plt.hist,'Age',alpha=.5,bins=20)
#grid.add_legend()

#grid = sns.FacetGrid(train_df,row='Embarked',size=2.2,aspect=1.6)
#grid.map(sns.pointplot,'Pclass','Sex','Survived',palette='deep')
#grid.add_legend()
#grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
#grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
#grid.add_legend()
#plt.show()


train_df = train_df.drop(['Ticket','Cabin'],axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df,test_df]

#print train_df.shape
#print test_df.shape
#print combine[0].shape
#print combine[1].shape

#print train_df.columns.values
#print test_df.columns.values


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print pd.crosstab(train_df['Title'],train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_maping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_maping)
    dataset['Title'] = dataset['Title'].fillna(0)

print train_df.head()

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print train_df.shape, test_df.shape


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)

print train_df.head()


guess_ages = np.zeros((2,3))


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

print train_df.head()


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16,'Age'] = 0
    dataset.loc[(dataset['Age'] > 16)&(dataset['Age'] <= 32) , 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64,'Age']

print train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print train_df.head()

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

print train_df.head()

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

print train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print train_df.head()

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print train_df.head(10)
print test_df.head(10)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

import re

from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import GridSearchCV # for Hyper parameter tuning
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn import metrics

UST2 = pd.read_csv('../final-1.csv')
print(UST2.head())
print(UST2.shape)

UST = pd.read_csv('../final-1.csv')
print(UST.head(10))
print(UST.shape)

# See the column data types and non-missing values
print(UST.info())
print(UST.describe())

# seems fine
print(UST['CLI'].mean(), UST['CLI'].median())

print(UST.columns)
print(UST.shape)

all_data = UST

traindf=UST

traindf = pd.get_dummies(traindf, columns = ["term","grade","sub_grade","emp_length","home_ownership","verification_status","loan_status","earliest_cr_line","last_credit_pull_d"],
                             prefix=["Owner","Account","Stg","SvcLn","StgNum","Acctype","Billngtype","UnSolProp","DelType"])

print(traindf.head())

# total of 100 records in dataset
# Divide the data set into two- 70% for train and 30% for test

train, test = train_test_split(traindf, test_size=0.3)

# Check the train test data set shape
train.shape, test.shape

# Add binary label column, modify test
trainBinary = []
testBinary = []

for num in np.random.randint(2, size=train.shape[0]):
    trainBinary.append(num)

for num in np.random.randint(2, size=test.shape[0]):
    testBinary.append(num)

train['binary'] = trainBinary
test['binary'] = testBinary

test.drop(['CLI'], axis = 1, inplace = True)

print(list(train.columns).index('binary'))
print(list(test.columns).index('binary'))

# Save to csv files for XGBoost
train.to_csv('xg_train_final.csv')
test.to_csv('xg_test_final.csv')

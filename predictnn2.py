#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:49:22 2017

@author: manish
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

def return_features(sample1, sample2):

    df = pd.read_csv(sample1)
    train = df[:]
 #   train = df[0:712]
  #  test =df [712:]
    
    df2 = pd.read_csv(sample2)
    #
    test = df2[:]
    
    test_y_pi = test.iloc[:,[0]]
    
    full_data = [train,test]
    
    for dataset in full_data:
        dataset['Name_Length'] = dataset['Name'].apply(len)
        
    for dataset in full_data:
        dataset['Has_Cabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    for dataset in full_data:
        dataset['IsAlone'] = dataset['FamilySize'].apply(lambda x: 1 if x==1 else 0)    
        
    for dataset in full_data:
        dataset['Sex'] = dataset['Sex'].apply(lambda x: 1 if x=='male' else 0)
    
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
        
    for dataset in full_data:
        dataset['Age'] = dataset['Age'].fillna(train['Age'].median())
        
    for dataset in full_data:
        emb_mapping = {'S' : 0, 'Q': 1, 'C' : 2}
        dataset['Embarked'] = dataset['Embarked'].map(emb_mapping)
        
    train['CategorialFare'] = pd.qcut(train['Fare'],5)
    train['CategorialAge'] = pd.qcut(train['Age'],4)
    
    for dataset in full_data:
        dataset.loc[ dataset['Fare'] <= 7.9, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.9) & (dataset['Fare'] <= 10.75), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 10.75) & (dataset['Fare'] <= 22.81), 'Fare'] = 2
        dataset.loc[(dataset['Fare'] > 22.81) & (dataset['Fare'] <= 41.58), 'Fare'] = 3
        dataset.loc[ dataset['Fare'] > 41.58, 'Fare'] = 4
        dataset['Fare'] = dataset['Fare'].astype(int)
        
        # Mapping Age
        dataset.loc[ dataset['Age'] <= 22, 'Age'] 					       = 0
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 28), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 36), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 80), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 80, 'Age'] = 4 ;
    
    drop_features = ['PassengerId','Name','Ticket','Cabin','Name_Length','FamilySize']
    drop_features_2 = ['CategorialFare','CategorialAge']
    train = train.drop(drop_features, axis= 1)
    train = train.drop(drop_features_2, axis= 1)
    test = test.drop(drop_features, axis=1)
    
#matrix train_x and matrix test_x are not equa;..rectify it    
    
    train_x = train.iloc[:,1:]
    train_y = train.iloc[:,[0]]
    test_x = test.iloc[:,0:]
 #   test_y = test.iloc[:,[0]]
    
    m= train_x.shape[0]
    full_data = pd.concat([train_x, test_x])
    
    full_data = onehotencoder.fit_transform(full_data).toarray()
    
    train_x = full_data[:m,:]
    test_x = full_data[m:,:]
    

    train_y = onehotencoder.fit_transform(train_y).toarray()
        
    return train_x, train_y, test_x, test_y_pi.values.ravel() #, test_y


#def confusion_matrix(train_x, train_y, test_x, test_y):
#    
#    classifier = SVC(kernel = 'rbf', random_state = 0)
#    classifier.fit(train_x, train_y)
#    y_pred = classifier.predict(test_x)
#    from sklearn.metrics import confusion_matrix
#    cm = confusion_matrix(test_y, y_pred)
#    return cm


if __name__ == '__main__':
    train_x, train_y, test_x , test_y_pi= return_features('train.csv','test.csv')
#   cm = confusion_matrix(train_x, train_y, test_x, test_y)
#    print(cm)
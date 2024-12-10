#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:48:18 2023

@author: francesco

In this script there is the prediction of Heckmatt score from precomputed texture 
features

A - class distribution visualization post PCA
B - classification
C - regression
D - look at misclassified

"""

import os
import json

from tqdm import tqdm 
import numpy as np

from scipy import stats
import pandas as pd

dataDir = '/media/francesco/DEV001/PROJECT-FSHD/'

## LOAD TEXTURE FROM JSON FILES

filename = os.path.join(dataDir, 'RESULTS', 'JSON', 'gtTextureMuscle_ClassHead.json')
with open(filename,'r') as f:
    textureGT = json.load(f)
    
## LOAD HECKMATT FROM JSON FILES

filename = os.path.join(dataDir, 'DATA', 'TABULAR', 'heckMapPlusCharacteristics.xlsx')
HeckMap = pd.read_excel(filename)
HeckMap = HeckMap.iloc[5:,:]

##     MAKE PREDICTION TABLE
ModelInput = []

for index, row in tqdm(HeckMap.iterrows()): # iter over subjects
    
    if not np.isnan(row['Code']):
        
        rowH = row.iloc[5:]
        sID = row['Code']
        
        for idRow, Hrow in rowH.iteritems(): # look into muscle-side of the specific subject
            
            temp = dict()
            
            temp['subj_muscle_side'] = '{:05d}_{}'.format(int(sID), idRow)
            temp['Heck_Score'] = Hrow
            
            for i in textureGT: # fint muscle-side-subject specific features
            
                if temp['subj_muscle_side'] in i['Image'][0]:
                # if temp['subj_muscle_side'] in i['Image']:
                    
                    temp['FSHD_age'] = row['FSHD_age']
                    temp['FSHD_BMI'] = row['FSHD_BMI']
                    temp['Sex'] = row['Sex']
                    
                    for k in i['features']:
                        temp[k] = np.mean(np.asarray(i['features'][k],dtype=np.float32))
        
        # temp = pd.Series(temp)
        # ModelInput = pd.concat([ModelInput, temp], ignore_index=True)
            ModelInput.append(temp)
        
ModelInput1 = pd.DataFrame(ModelInput)
ModelInput1 = ModelInput1.set_index('subj_muscle_side')

## A - DIMENSIONALITY REDUCTION AND CLASS VISUALIZATION

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Dataset = ModelInput1.to_numpy()
Dataset = Dataset[~np.isnan(Dataset).any(axis=1)]

X = Dataset[:,1:]                    
Y = Dataset[:,0].astype(np.uint8)-1

sc = StandardScaler()
sc.fit(X)
Xsc = sc.transform(X)


import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

X, y = Xsc, Y

features = ModelInput1.columns.to_list()[1:]

# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# enumerate splits
outer_results = list()
predicted = []
reals = []

fimpAll = np.zeros((96))


for train_ix, test_ix in cv_outer.split(X,y):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    
   
    # define the model
    model = XGBRegressor(n_estimators=100,
                         max_depth=6, eta=0.15, colsample_bytree=0.7)
    

    model.fit(X_train, y_train)

    # evaluate model on the hold out dataset
    yhat = model.predict(X_test)
    
    # evaluate the model
    mae = mean_absolute_error(y_test, yhat)
    
    # store the result
    outer_results.append(mae)
    predicted.append(yhat)
    reals.append(y_test)
    
    fimp = model.feature_importances_
    fimpAll += fimp
    sorted_idx = model.feature_importances_.argsort()[::-1][:9]
    
    print("Most 10 important features :\n")
    
    for ww in sorted_idx:
        print('\n\t {} :: {}'.format(features[ww], fimp[ww]))
        
    # # report progress
    print('>mae=%.3f\n' % (mae))
 
# summarize the estimated performance of the model
print('mae: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))


CLASSES = [0,1,2,3]

from itertools import chain

yts1 = list(chain.from_iterable(reals))
pts1 = list(chain.from_iterable(predicted))

import seaborn as sns
sns.scatterplot(x=pts1, y=yts1);

from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import matthews_corrcoef, log_loss, cohen_kappa_score

cm_all = confusion_matrix(yts1, pts1)
cm_all_norm = confusion_matrix(yts1, pts1, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm_all,
                              display_labels=CLASSES)
disp.plot()
plt.show()

print('\n kappa = {:05f}'.format(cohen_kappa_score(yts1, pts1)))

fimpAll /= 10

sorted_idx = fimpAll.argsort()[::-1][:9]

print("Most 10 important features :\n")

columns = []

for ww in sorted_idx:
    print('\n\t {} :: {}'.format(features[ww], fimpAll[ww]))
    columns.append(features[ww])
    




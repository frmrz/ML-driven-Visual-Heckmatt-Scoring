#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:58:59 2023

@author: francesco
"""

import os
import json

from tqdm import tqdm 
import numpy as np

from scipy import stats
import pandas as pd

dataDir = '/media/francesco/DEV001/PROJECT-FSHD/'

## LOAD TEXTURE FROM JSON FILES :::: Image to Image

filename = os.path.join(dataDir, 'RESULTS', 'JSON', 'gtTextureMuscle.json')
# filename = os.path.join(dataDir, 'RESULTS', 'JSON', 'gtTextureMuscle_ClassHead.json')
# filename = os.path.join(dataDir, 'RESULTS', 'JSON', 'gtTexture_ClassHead_fullImage.json')

with open(filename,'r') as f:
    textureGT = json.load(f)
    
filename = os.path.join(dataDir, 'RESULTS', 'JSON', 'predTextureMuscle_ClassHead.json')

with open(filename,'r') as f:
    texturePred = json.load(f)
    
# filename = os.path.join(dataDir, 'DATA', 'TABULAR', 'heckMap.xlsx')
filename = os.path.join(dataDir, 'DATA', 'TABULAR', 'heckMapPlusCharacteristics.xlsx')

HeckMap = pd.read_excel(filename)
HeckMap = HeckMap.iloc[5:,:]

##     MAKE PREDICTION TABLE

## Rows : subjects - Muscle combo (From HeckMap)
## Cols : (Predictors) Feat1 ... -> ... FeatN // (Target) HeckScore

# Create dict Code-Target
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

##     PREPARE DATA FORM CLASSIFIER

from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import matthews_corrcoef, log_loss, cohen_kappa_score
from sklearn.preprocessing import robust_scale, scale
from sklearn.preprocessing import minmax_scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import class_weight
import numpy as np
import cleanlab
from cleanlab.classification import CleanLearning
from cleanlab.benchmarking import noise_generation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from numpy.random import multivariate_normal

import gc
import matplotlib.pyplot as plt

SEED=5

Dataset = ModelInput1.to_numpy()
features = ModelInput1.columns.to_list()[1:]

Dataset = Dataset[~np.isnan(Dataset).any(axis=1)]

X = Dataset[:,1:]                    
X = minmax_scale(X,axis=0)               
Y = Dataset[:,0].astype(np.uint8)-1

Dataset = np.column_stack((Y,X))

yourFavoriteModel = XGBClassifier()
cl = cleanlab.classification.CleanLearning(yourFavoriteModel, seed=SEED)

# Fit model to messy, real-world data, automatically training on cleaned data.
_ = cl.fit(X, Y)

issues = CleanLearning(yourFavoriteModel, seed=SEED).find_label_issues(X, Y)

# CleanLearning can train faster if issues are provided at fitting time.
cl.fit(X, Y, label_issues=issues)

cleanlab.dataset.find_overlapping_classes(
    labels=Y,
    confident_joint=cl.confident_joint,  # cleanlab uses the confident_joint internally to quantify label noise (see cleanlab.count.compute_confident_joint)
    class_names=['1','2','3','4'],
)

## TRAIN MODEL

n_splits = 5
kf = KFold(n_splits=n_splits, random_state=46, shuffle=True)

yts = []
pts = []
    
fimpAll = np.zeros((96))

for jj, (train_index, test_index) in enumerate(kf.split(X)):
    
    
    print("Fitting fold {}\n".format(jj+1))
    
    train_features = X[train_index]
    train_target   = Y[train_index]
    
    test_features = X[test_index]
    test_target   = Y[test_index]
    
    yts.append(test_target)

    pts_temp = []
    
    data=train_features
    labels=train_target
    
    test_data=test_features
    test_labels=test_target
    
    yourFavoriteModel = XGBClassifier()

    # CleanLearning: Machine Learning with cleaned data (given messy, real-world data)
    cl = cleanlab.classification.CleanLearning(yourFavoriteModel, seed=SEED)

    # Fit model to messy, real-world data, automatically training on cleaned data.
    _ = cl.fit(data, labels)

    # See the label quality for every example, which data has issues, and more.
    cl.get_label_issues().head()

    # For comparison, this is how you would have trained your model normally (without Cleanlab)
    yourFavoriteModel = LogisticRegression(verbose=0, random_state=SEED)
    yourFavoriteModel.fit(data, labels)
    print(f"Accuracy using yourFavoriteModel: {yourFavoriteModel.score(test_data, test_labels):.0%}")

    # But CleanLearning can do anything yourFavoriteModel can do, but enhanced.
    # For example, CleanLearning gives you predictions (just like yourFavoriteModel)
    # but the magic is that CleanLearning was trained as if your data did not have label errors.
    print(f"Accuracy using yourFavoriteModel (+ CleanLearning): {cl.score(test_data, test_labels):.0%}")


    # model = XGBClassifier(use_rmm=True,verbosity=0,nthread=16,
    #                       eta=0.3, # learning rate
    #                       gamma=0, # Minimum loss reduction required to make a further partition on a leaf node of the tree
    #                       max_depth=6, # Maximum depth of a tree
    #                       min_child_weight=1, # Minimum sum of instance weight (hessian) needed in a child
    #                         # subsample = 1, # Subsample ratio of the training instances
    #                         # sampling_method = 'uniform', # method to use to sample the training instances
    #                         # tree_method = 'gpu_hist',
    #                         # process_type = 'default',
    #                         # predictor = 'gpu_predictor',
    #                         # objective = 'multi:softmax', # reg:gamma
    #                         # eval_metric = ['merror','mlogloss'] # reg:gamma
    #                       )
    model = XGBClassifier()
    
    # classes_weights = class_weight.compute_sample_weight(
    #     class_weight='balanced',
    #     y=train_target
    # )
    
    # model.fit(train_features, train_target, sample_weight=classes_weights)
    model.fit(train_features, train_target)
            
    test_pred = model.predict(test_features)
    pts_temp.append(test_pred)
    
    fimp = model.feature_importances_
    fimpAll += fimp
    sorted_idx = model.feature_importances_.argsort()[::-1][:9]
    
    print("Most 10 important features :\n")
    
    for ww in sorted_idx:
        print('\n\t {} :: {}'.format(features[ww], fimp[ww]))

    del train_features, train_target, test_features, test_target, model
    
    gc.collect()

    pts.append([np.mean(np.asarray(pts_temp),axis=0).round()])
    

yts1 = np.concatenate((yts[0],yts[1],yts[2],yts[3],yts[4]),axis=0)
pts1 = np.concatenate((pts[0][0],pts[1][0],pts[2][0],pts[3][0],pts[4][0]),axis=0)

CLASSES = [0,1,2,3]

cm_all = confusion_matrix(yts1, pts1)
cm_all_norm = confusion_matrix(yts1, pts1, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm_all,
                              display_labels=CLASSES)
disp.plot()
plt.show()

print('\n kappa = {:05f}'.format(cohen_kappa_score(yts1, pts1)))

fimpAll /= 5

sorted_idx = fimpAll.argsort()[::-1][:9]

print("Most 10 important features :\n")

columns = []

for ww in sorted_idx:
    print('\n\t {} :: {}'.format(features[ww], fimpAll[ww]))
    columns.append(features[ww])
    
## PLOT FEATURES

BestFeaturesHeck1 = X[Y == 0,:]
BestFeaturesHeck1 = BestFeaturesHeck1[:,sorted_idx]    
BestFeaturesHeck1 = pd.DataFrame(BestFeaturesHeck1, columns=columns, index=range(len(BestFeaturesHeck1)))
    
BestFeaturesHeck2 = X[ Y == 1,:]
BestFeaturesHeck2 = BestFeaturesHeck2[:,sorted_idx]    
BestFeaturesHeck2 = pd.DataFrame(BestFeaturesHeck2, columns=columns, index=range(len(BestFeaturesHeck2)))

BestFeaturesHeck3 = X[ Y == 2,:]
BestFeaturesHeck3 = BestFeaturesHeck3[:,sorted_idx]    
BestFeaturesHeck3 = pd.DataFrame(BestFeaturesHeck3, columns=columns, index=range(len(BestFeaturesHeck3)))

BestFeaturesHeck4 = X[Y == 3,:]
BestFeaturesHeck4 = BestFeaturesHeck4[:,sorted_idx]        
BestFeaturesHeck4 = pd.DataFrame(BestFeaturesHeck4, columns=columns, index=range(len(BestFeaturesHeck4)))

import seaborn as sns

Dataset_for_display = pd.DataFrame(Dataset, columns = ModelInput1.columns.to_list(), index=range(len(Dataset)))
Dataset_for_display["Heck_Score"] = Dataset_for_display["Heck_Score"].astype("category")

# sns.barplot(data=Dataset_for_display, y=columns[0], x="Heck_Score")    
# sns.barplot(data=Dataset_for_display, y=columns[1], x="Heck_Score")    
# sns.barplot(data=Dataset_for_display, y=columns[2], x="Heck_Score")    
# sns.barplot(data=Dataset_for_display, y=columns[3], x="Heck_Score")    
# sns.barplot(data=Dataset_for_display, y=columns[4], x="Heck_Score")    
# sns.barplot(data=Dataset_for_display, y=columns[5], x="Heck_Score")    
# sns.barplot(data=Dataset_for_display, y=columns[6], x="Heck_Score")    
# sns.barplot(data=Dataset_for_display, y=columns[7], x="Heck_Score")    
# sns.barplot(data=Dataset_for_display, y=columns[8], x="Heck_Score")    
    
    
## PRINCIPAL COMPONENT ANALYSIS

from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as io
io.renderers.default='browser'

pca = PCA(n_components=3)
pca.fit(X)

print(pca.components_)
print(pca.explained_variance_)

X_pca = pca.transform(X)

df = np.column_stack((Y,X_pca))
df = pd.DataFrame(df,
                  columns = ['HS','PCA1','PCA2','PCA3'],
                  index=range(len(df)))
df["HS"] = df["HS"].astype("category")

fig = px.scatter_3d(df,
                    x='PCA1',
                    y='PCA2',
                    z='PCA3',
                    color='HS',
                    size_max=1)

fig.show()

X_best = X[:,sorted_idx]
    
pca_best = PCA(n_components=3)
pca_best.fit(X_best)

X_pca_best = pca_best.transform(X_best)

df = np.column_stack((Y,X_pca_best))
df = pd.DataFrame(df,
                  columns = ['HS','PCA1','PCA2','PCA3'],
                  index=range(len(df)))
df["HS"] = df["HS"].astype("category")

fig = px.scatter_3d(df,
                    x='PCA1',
                    y='PCA2',
                    z='PCA3',
                    color='HS',
                    opacity = 0.7,
                    size_max=0.1)
fig.show()

## REPEAT TRAIN WITH PCA

pca = PCA(n_components=10)
pca.fit(X)

X_pca = pca.transform(X)

yts = []
pts = []

for jj, (train_index, test_index) in enumerate(kf.split(X)):
    
    
    print("Fitting fold {}\n".format(jj+1))
    
    train_features = X_pca[train_index]
    train_target   = Y[train_index]
    
    test_features = X_pca[test_index]
    test_target   = Y[test_index]
    
    yts.append(test_target)

    pts_temp = []
    

    model = XGBClassifier()

    model.fit(train_features, train_target)
            
    test_pred = model.predict(test_features)
    pts_temp.append(test_pred)
    

    del train_features, train_target, test_features, test_target, model
    
    gc.collect()

    pts.append([np.mean(np.asarray(pts_temp),axis=0).round()])
    

yts1 = np.concatenate((yts[0],yts[1],yts[2],yts[3],yts[4]),axis=0)
pts1 = np.concatenate((pts[0][0],pts[1][0],pts[2][0],pts[3][0],pts[4][0]),axis=0)

CLASSES = [0,1,2,3]

cm_all = confusion_matrix(yts1, pts1)
cm_all_norm = confusion_matrix(yts1, pts1, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm_all,
                              display_labels=CLASSES)
disp.plot()
plt.show()
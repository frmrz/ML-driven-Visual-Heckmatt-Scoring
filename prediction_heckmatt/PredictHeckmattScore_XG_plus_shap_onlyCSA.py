#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:34:51 2023

@author: francesco
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend like 'Agg'
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image

import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
    
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import combinations

# Import necessary libraries
from xgboost import XGBClassifier
from shap import Explanation

# Importing shap library
import shap


def generate_shap_plots_for_class(dfOut_skf, X_test, dataDir, result_dir, explainer, feature_names, observations, ii, class_num,seed=123456, link_f='logit'):
   
    assert class_num in [1, 2, 3], "class_num must be 1, 2, or 3."
    
    plt.rcParams['figure.dpi'], plt.rcParams['font.size'] = 300, 12
    
    correct_class = dfOut_skf.loc[(dfOut_skf['manual_h_score'] == class_num-1) & (dfOut_skf['predicted_h_score'] == class_num-1)]
    
    random_index = correct_class.sample(1, random_state=seed).index
    random_index_tuple = tuple(random_index.values[0])
    random_index_idx = X_test.index.get_loc(random_index_tuple)
    
    subject_number, muscle_number, side_number = random_index[0]
    filename = f'{str(subject_number).zfill(5)}_{str(muscle_number).zfill(3)}_{str(side_number).zfill(2)}_1.png'
    # img_path = os.path.join(dataDir, 'DATA', 'DL_FSHD_reLabel', 'modified_images_gauss', filename)
    img_path = os.path.join(dataDir, 'DATA', 'DL_FSHD_debug', filename)
    img = plt.imread(img_path)
    
    img_filename = f'Image{class_num}_{ii}.png'
    imgPIL = Image.open(img_path)
    
    if link_f == 'logit':
        imgPIL.save(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', img_filename))
    else:
        imgPIL.save(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', img_filename))
    
    # Beeswarm plot to identify top 10 features
    shp = explainer(observations)[...,class_num-1]
    shp.feature_names = feature_names
    
    plt.figure()
    shap.plots.beeswarm(shp, max_display=11, show=False) # to plot this shap_values = explainer(X_train) but check X_test
    plt.tight_layout()
        
    beeswarm_plot_filename = f'ShapBeeswarmPlot_{class_num}_{ii}.png'
    beeswarm_plot_filename_pdf = f'ShapBeeswarmPlot_{class_num}_{ii}.pdf'
    
    if link_f == 'logit':
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', beeswarm_plot_filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', beeswarm_plot_filename_pdf), bbox_inches='tight')  # Save as PDF for vector format
    else:
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', beeswarm_plot_filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', beeswarm_plot_filename_pdf), bbox_inches='tight')
    
    plt.clf()

    # Extract top 10 features - This step is a placeholder. Actual extraction depends on external methods or manual selection

    top_features_indices = np.argsort(-np.abs(shp.values).mean(axis=0))[:10]
    top_features = np.array(feature_names)[top_features_indices]

    # Decision plot with top 10 features
    top_features_shap_values = shp[random_index_idx, top_features_indices]
    
    # calculate the overall effect of the non-top features
    other_features_shap_values = shp[random_index_idx, [i for i in range(len(feature_names)) if i not in top_features_indices]]
    other_features_effect = np.sum(other_features_shap_values.values)
    N = len(other_features_shap_values.values)
    
    # create list of indexes for the top features from top_features_indices + 1 length to 0
    top_features_order = list(range(len(top_features_indices), -1, -1))
    
    # append the overall effect of the non-top features to the top features values
    top_features_shap_values.values = np.append(top_features_shap_values.values, other_features_effect)
    
    # append the overall effect of the non-top features to the top features names, with the name "Sum of N Other features"
    top_features = np.append(top_features, f"Sum of {N} Other features")
    
    # apply softmax transform to explainer.expected_value
    exp_val_prob = np.exp(explainer.expected_value) / np.sum(np.exp(explainer.expected_value))
    
    # Compute SHAP values for all classes
    shap_values = explainer(observations)
    
    # extract the SHAP values of the random index
    top_shap_values_idx = shap_values[:,top_features_indices,...]
    
    # calculate the overall effect of the non-top features over all classes
    other_features_shap_values = shap_values[:, [i for i in range(len(feature_names)) if i not in top_features_indices],...]
    other_features_effect = np.sum(other_features_shap_values.values, axis=1)
    
    # expand axis 1 of the other_features_effect to match the shape of top_shap_values_idx
    other_features_effect = np.expand_dims(other_features_effect, axis=1)
    
    top_shap_values_idx.values = np.append(top_shap_values_idx.values, other_features_effect, axis=1)
        
    # organize the top_features_shap_values_idx as if there was one model for each class
    top_shap_1 = top_shap_values_idx[...,0]
    top_shap_2 = top_shap_values_idx[...,1]
    top_shap_3 = top_shap_values_idx[...,2]    
        
    shap1 = top_shap_1.values
    shap2 = top_shap_2.values
    shap3 = top_shap_3.values
        
    base1 = top_shap_1.base_values.reshape(-1, 1)
    base2 = top_shap_2.base_values.reshape(-1, 1)
    base3 = top_shap_3.base_values.reshape(-1, 1)
    
    labels = ['Normal', 'Uncertain', 'Abnormal']
        
    shap_list = [shap1, shap2, shap3]
    base_values = [base1, base2, base3]
    
    shap.multioutput_decision_plot(base_values, 
                shap_list, 
                random_index_idx,
                feature_names=top_features,
                feature_order=top_features_order, 
                legend_labels=labels,
                link=link_f,
                legend_location='lower right',
                show=False)
    
    plt.tight_layout()
    
    decision_plot_filename = f'ShapMultiDecisionPlotCorrectClass{class_num}_{ii}.png'
    decision_plot_filename_pdf = f'ShapMultiDecisionPlotCorrectClass{class_num}_{ii}.pdf'
    
    if link_f == 'logit':
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', decision_plot_filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', decision_plot_filename_pdf), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', decision_plot_filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', decision_plot_filename_pdf), bbox_inches='tight')

    plt.clf()
    
    # Composite image plot
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.1) 
    
    ax[0].imshow(img, cmap='gray')
    ax[0].axis('off')
    
    if link_f == 'logit':
        ax[1].imshow(plt.imread(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', beeswarm_plot_filename)))
    else:   
        ax[1].imshow(plt.imread(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', beeswarm_plot_filename)))
    ax[1].axis('off')
    
    if link_f == 'logit':
        ax[2].imshow(plt.imread(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', decision_plot_filename)))
    else:
        ax[2].imshow(plt.imread(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', decision_plot_filename)))
    ax[2].axis('off')
    
    plt.tight_layout()
    
    final_plot_filename_png = f'CorrectClass{class_num}_{ii}.png'
    final_plot_filename_pdf = f'CorrectClass{class_num}_{ii}.pdf'
    
    if link_f == 'logit':
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', final_plot_filename_png), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', final_plot_filename_pdf), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', final_plot_filename_png), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', final_plot_filename_pdf), bbox_inches='tight')
        
    plt.clf()


# Create a crosstab function for plotting
def plot_crosstab(train_set, test_set, fold_num):
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Plot training set
    crosstab_train = pd.crosstab(train_set['muscleN'], train_set['manual_h_score'])
    crosstab_train.rename(index=code_to_class).plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title('Fold {} - Training Set'.format(fold_num))
    axes[0].set_xlabel('Muscle Type')
    axes[0].set_ylabel('Count')

    # Plot test set
    crosstab_test = pd.crosstab(test_set['muscleN'], test_set['manual_h_score'])
    crosstab_test.rename(index=code_to_class).plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Fold {} - Test Set'.format(fold_num))
    axes[1].set_xlabel('Muscle Type')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()
    
# Define a custom function to calculate the mean of features with less than 10% variation
def mean_features_with_less_variation(group):
    feature_columns = group["features_img_gt"].iloc[0].keys()
    mean_features = {}

    for feature in feature_columns:
        values = [float(entry[feature]) for entry in group["features_img_gt"]]
        variation = max(values) - min(values)
        # if variation / abs(np.mean(values) + 1e-10) < 0.25:
        if variation / abs(np.mean(values) + 1e-10) < 0.5:
            mean_features[feature] = sum(values) / len(values)
        else:
            mean_features[feature] = sum(values) / len(values) # this is done because we want to keep the feature anyway, the check on variation is perfromed using "nan" version of the function

    return pd.Series(mean_features)

def mean_features_with_less_variation_nan(group):
    feature_columns = group["features_img_gt"].iloc[0].keys()
    mean_features = {}

    for feature in feature_columns:
        values = [float(entry[feature]) for entry in group["features_img_gt"]]
        variation = max(values) - min(values)
        # if variation / abs(np.mean(values) + 1e-10) < 0.25:
        if variation / abs(np.mean(values) + 1e-10) < 0.5:
            mean_features[feature] = sum(values) / len(values)
        else:
            mean_features[feature] = np.nan

    return pd.Series(mean_features)

dataDir = '/media/francesco/DEV001/PROJECT-FSHD/'
filename = os.path.join(dataDir, 'DATA', 'TABULAR', 'heckMapPlusCharacteristics.xlsx')
HeckMap = pd.read_excel(filename)
HeckMap = HeckMap.iloc[5:,:]

filename = "/media/francesco/DEV001/PROJECT-FSHD/RESULTS/EXCEL/segmentation_summary_knet_swin_mod_pred.json"

data = json.load(open(filename, "r"))
df = pd.DataFrame.from_dict(data)

del data
import gc
gc.collect()

df_head = df.head()

# Creating a dictionary that represents the hash table
class_to_code = {
    'Biceps_brachii': '001',
    'Deltoideus': '002',
    'Depressor_anguli_oris': '003',
    'Digastricus': '004',
    'Extensor_digitorum_brevis': '005',
    'Flexor_carpi_radialis': '006',
    'Flexor_digitorum_profundus': '007',
    'Gastrocnemius_medial_head': '008',
    'Geniohyoideus': '009',
    'Levator_labii_superior': '010',
    'Masseter': '011',
    'Mentalis': '012',
    'Orbicularis_oris': '013',
    'Peroneus_tertius': '014',
    'Rectus_abdominis': '015',
    'Rectus_femoris': '016',
    'Temporalis': '017',
    'Tibialis_anterior': '018',
    'Trapezius': '019',
    'Vastus_lateralis': '020',
    'Zygomaticus': '021'
    }

code_to_class_original = {
    '001': 'BB',
    '002': 'DEL',
    '003': 'DA',
    '004': 'DIG',
    '005': 'EDB',
    '006': 'FCR',
    '007': 'FDP',
    '008': 'GM',
    '009': 'GH',
    '010': 'LLS',
    '011': 'MAS',
    '012': 'MNT',
    '013': 'OO',
    '014': 'PT',
    '015': 'RA',
    '016': 'RF',
    '017': 'TEM',
    '018': 'TA',
    '019': 'TRAP',
    '020': 'VL',
    '021': 'ZYG'
}

class_to_code_HZ = {
    'Biceps': '001',
    'GM': '008',
    'Rectusab': '015',
    'RF': '016',
    'TA': '018',
    'Trap': '019',
    'VL': '020'
    }

side_to_code = {
    'L': '01',
    'R': '00'
}
# Creating the new 'muscle_code' column
df['muscle_code'] = df['class_gt'].map(class_to_code)

# Merging the two dataframes
HeckMap['Code'] = HeckMap['Code'].apply(lambda x: str(int(float(x))).zfill(5) if pd.notnull(x) and x != '' else '')

HeckMap['Code'] = HeckMap['Code'].astype(str)
HeckMap['Sex'] = HeckMap['Sex'].astype(str)
HeckMap['FSHD_age'] = HeckMap['FSHD_age'].astype(str)
HeckMap['FSHD_BMI'] = HeckMap['FSHD_BMI'].astype(str)

##########
###### CREATE DATASET
##########   

df1 = pd.merge(df, HeckMap[['Code', 'Sex', 'FSHD_age', 'FSHD_BMI']], left_on='subject', right_on='Code', how='left')

# Dropping the 'code' column as it's not required anymore
df1 = df1.drop('Code', axis=1)

# Rename columns from HeckMap in df
df1.rename(columns={'Sex': 'sex', 'FSHD_age': 'age', 'FSHD_BMI': 'bmi'}, inplace=True)

# Ensure 'muscle_code' and 'side' are strings
df1['muscle_code'] = df1['muscle_code'].astype(str)
df1['side'] = df1['side'].astype(str)

# Create a new column 'column_name' that contains the dynamic column names
df1['muscle_side'] = df1['muscle_code'] + '_' + df1['side']

# Iterate over the rows of the dataframe
for idx, row in df1.iterrows():
    # Get the dynamic column name
    column_name = row['muscle_side']
    subject_name = row['subject']
    # If the dynamic column name exists in the dataframe, assign the value to 'manual_h_score'
    if column_name in HeckMap.columns and subject_name in HeckMap['Code'].values:
        find_idx = HeckMap['Code'].loc[lambda x: x==subject_name].index[0]
        df1.loc[idx, 'manual_h_score'] = HeckMap.loc[find_idx, column_name]
    else:
        df1.loc[idx, 'manual_h_score'] = np.nan
        
##########
###### MAKE DATAFRAME TO PREDICT HECKMATT SCORE
##########    

df_hPred = df1.copy()

# Replace 'mask not found' with NaN
df_hPred.replace('mask not found', np.nan, inplace=True)

df_hPred = df_hPred.dropna(axis=0)

# Standardize the numerical features
scaler = StandardScaler()

# Convert 'class_gt' and 'class_pred' to numeric, converting errors to NaN
df_hPred['subject'] = pd.to_numeric(df_hPred['subject'], errors='coerce')
df_hPred['age'] = pd.to_numeric(df_hPred['age'], errors='coerce')
df_hPred['bmi'] = pd.to_numeric(df_hPred['bmi'], errors='coerce')
df_hPred['muscleN'] = pd.to_numeric(df_hPred['muscle'], errors='coerce')

# Get the dataframe for GT features
df_gt = df_hPred.loc[:,['subject','muscle','side','age','bmi','sex','muscleN','features_img_gt','features_img_gt_not','manual_h_score']]

# Group the DataFrame and calculate the mean of features with less than 10% variation 
# Nan are used to flag the features with more than N spatial variation
grouped_df = df_gt.groupby(['subject', 'muscle','side']).apply(mean_features_with_less_variation).reset_index()
grouped_df_nan = df_gt.groupby(['subject', 'muscle','side']).apply(mean_features_with_less_variation_nan).reset_index()

# Calculate the threshold value
threshold = grouped_df.shape[0] * 0.1

# Exclude columns with NaN count higher than the threshold
filtered_df = grouped_df.loc[:, grouped_df_nan.isna().sum() <= threshold]

# Exclude columns 'subject', 'muscle', and 'side'
filtered_df = filtered_df.drop(['subject', 'muscle', 'side'], axis=1)

# Transform each row into a dictionary
data_dict = filtered_df.to_dict(orient='records')

df_hPred_group = df_hPred.groupby(['subject', 'muscle','side']).agg('first').reset_index()
df_hPred_group["manual_h_score"] = df_hPred_group["manual_h_score"].replace(4,3)

df_hPred_group["features_img_gt"] = data_dict

# load merged_df_out.csv
df_H_Z = pd.read_csv(os.path.join(dataDir, 'DATA', 'TABULAR', 'merged_df_out.csv'))

# change values of 'Muscle' column in df_H_Z using class_to_code_HZ
df_H_Z['Muscle'] = df_H_Z['Muscle'].map(class_to_code_HZ)

# change values of 'Side' column in df_H_Z using side_to_code
df_H_Z['Side'] = df_H_Z['Side'].map(side_to_code)

# find code-muscle-side combinations that are in df_H_Z but not in df_hPred_group
df_H_Z['muscle_side'] = df_H_Z['Muscle'] + '_' + df_H_Z['Side']
df_H_Z = df_H_Z.dropna(axis=0)

# rename 'Code' column to 'subject'
df_H_Z.rename(columns={'Code': 'subject'}, inplace=True)
# rename 'Muscle' column to 'muscle'
df_H_Z.rename(columns={'Muscle': 'muscle'}, inplace=True)
# rename 'Side' column to 'side'
df_H_Z.rename(columns={'Side': 'side'}, inplace=True)

# Create a multi-index based on the three columns in both dataframes
index_cols = ['subject', 'muscle', 'side']
df_hPred_group = df_hPred_group.set_index(index_cols)
df_B_indexed = df_H_Z.set_index(index_cols)

# Find entries in B that are not in A
entries_not_in_A = df_B_indexed[~df_B_indexed.index.isin(df_hPred_group.index)]

# remove entries_not_in_A from df_H_Z
df_H_Z_1 = df_B_indexed[~df_B_indexed.index.isin(entries_not_in_A.index)]

# on df_hPred_group, set the index to be the same as df_H_Z_1 and keep only manual_h_score
h_dfhpred1 = df_hPred_group.loc[df_H_Z_1.index, 'manual_h_score']
h_dfhz1 = df_H_Z_1['H']

# change h_dfhpred1 series name to 'H'
h_dfhpred1.name = 'H'

# Create a boolean mask indicating where the values are different
mask = h_dfhpred1 != h_dfhz1

# Use the mask to select the differing entries
differences = h_dfhpred1[mask]

print("Entries that are different between the two Pandas Series:")
print(differences)

##########
### Standardize features
################

result_dir = '/media/francesco/DEV001/PROJECT-FSHD/RESULTS/PAPER/'
excel_dir = '/media/francesco/DEV001/PROJECT-FSHD/RESULTS/EXCEL/'

feat_names_gt = list(data_dict[0].keys())

df_feat_gt_sc = pd.DataFrame(data=data_dict, columns=feat_names_gt)

scaled_features = df_feat_gt_sc

# Create a new DataFrame with the scaled features
scaled_df = pd.DataFrame(scaled_features, columns=feat_names_gt)

##### FEATURE SELECTION #####

# # Delete features with low variance
# from sklearn.feature_selection import VarianceThreshold

# # Create a VarianceThreshold feature selector
# selector = VarianceThreshold(threshold=0.1)

# # Fit the selector to the scaled DataFrame
# selector.fit(scaled_df)

# # Get the indices of the features that are being kept
# kept_features = selector.get_support(indices=True)

# # Get the names of the kept features
# kept_features_names = [feat_names_gt[i] for i in kept_features]

# # Keep only the features with high variance
# scaled_df = scaled_df[kept_features_names]

# feat_names_gt = kept_features_names

# # Calculate the correlation matrix
# correlation_matrix = scaled_df.corr()

# # Plot the correlation matrix
# plt.figure(figsize=(14, 10))
# sns.heatmap(correlation_matrix, cmap='RdBu', center=0)
# plt.title('Correlation Matrix')
# plt.savefig(os.path.join(result_dir, 'CorrCSA_CSAonly.png'), bbox_inches='tight')

# # save as svg
# plt.savefig(os.path.join(result_dir, 'CorrCSA_CSAonly.svg'), bbox_inches='tight')
# plt.show()


# # Find features with correlation coefficient greater than 0.95
# high_correlation = np.where(correlation_matrix > 0.75)

# # Create a list to store the features to be removed
# features_to_remove = []

# # Iterate over the correlation matrix
# for i, j in zip(*high_correlation):
#     # Exclude diagonal elements and duplicates
#     if i != j and i < j:
#         feature_i = feat_names_gt[i]
#         feature_j = feat_names_gt[j]
            
#         # Calculate the average correlation of each feature with other features
#         avg_corr_i = np.mean(correlation_matrix.iloc[i])
#         avg_corr_j = np.mean(correlation_matrix.iloc[j])
        
#         # Retain the feature with the lowest average correlation
#         if avg_corr_i < avg_corr_j:
#             features_to_remove.append(feature_j)
#         else:
#             features_to_remove.append(feature_i)
            
# # get unique values of features_to_remove
# features_to_remove = list(set(features_to_remove))

# # Exclude the features with high correlation from the DataFrame
# filtered_df = scaled_df.drop(columns=features_to_remove)


##################
##### CREATE X AND Y
##################

dfX = df_hPred_group.loc[:,['manual_h_score']]
dfX["manual_h_score"] = dfX["manual_h_score"].astype("category")

dfOut = dfX.copy()

# set index og filtered_df and filtered_df_not to be the same as dfX
filtered_df = scaled_df

filtered_df.index = dfX.index

# Concatenate the new DataFrame with the original one

dfX = pd.concat([dfX, filtered_df], axis=1)
tp = dfX.dtypes

# create X to have only the features
X = dfX.drop(columns=['manual_h_score'])

# create Y to have only the target, turn it to uint8 and subtract 1 to have the classes starting from 0
Y = dfX.loc[:,['manual_h_score']]
Y = Y.astype('uint8') - 1

tp = Y.dtypes

# create the 'groups' series using the first item of the dfX index
groups = dfX.index.get_level_values(0)
dfX.to_csv(os.path.join(excel_dir, 'preProcessedDF_CSAonly.csv'),index=False)
    
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import combinations

# Import necessary libraries
from xgboost import XGBClassifier
from shap import Explanation

# Importing shap library
import shap

# Creating a dictionary that represents the hash table
class_to_code = {
    'Biceps_brachii': 1,
    'Gastrocnemius_medial_head': 8,
    'Rectus_abdominis': 15,
    'Rectus_femoris': 16,
    'Tibialis_anterior': 18,
    'Trapezius': 19,
    'Vastus_lateralis': 20
}
code_to_class = {v: k for k, v in class_to_code.items()}

n_classes = 3

# Preprocessing for numerical and categorical features
num_features = list(X.columns)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features)])

# Replace LogisticAT with XGBClassifier in your pipeline
clf = Pipeline(steps=[

                        ('scaler', StandardScaler()),
                      
                      ('feature_selection', SelectFromModel(
                                                XGBClassifier(use_label_encoder=False,
                                                    device='cpu',
                                                    eta=0.2,
                                                    max_depth=4,
                                                    gamma=0.2,
                                                    min_child_weight=1,
                                                    subsample=1,
                                                    colsample_bytree=1,
                                                    n_estimators=100,
                                                    objective='multi:softprob',
                                                    eval_metric='auc',
                                                   ), prefit=False, threshold='median')),
                      
                      ('classifier', XGBClassifier(use_label_encoder=False,
                                                    device='cpu',
                                                    eta=0.2,
                                                    max_depth=4,
                                                    gamma=0.2,
                                                    min_child_weight=1,
                                                    subsample=1,
                                                    colsample_bytree=1,
                                                    n_estimators=100,
                                                    objective='multi:softprob',
                                                    eval_metric='auc',
                                                   ))
                ])

clf_min = Pipeline(steps=[('scaler', StandardScaler())])

# Create StratifiedKFold object
skf = StratifiedGroupKFold(n_splits=10)

Y_test_all = []
Y_pred_all = []
Y_pred_prob_all = []
features_all = []

mean_tpr_ovr = dict()
mean_auc_ovr = dict()
std_auc_ovr = dict()
mean_tpr_ovo = dict()
mean_auc_ovo = dict()
std_auc_ovo = dict()

tprs_ovr = dict()
aucs_ovr = dict()
tprs_ovo = dict()
aucs_ovo = dict()

mean_fpr_ovr = np.linspace(0, 1, 100)  # fixed FPR values for interpolation
mean_fpr_ovo = np.linspace(0, 1, 100)  # fixed FPR values for interpolation

for i in range(n_classes):
    tprs_ovr[i] = []
    aucs_ovr[i] = []

for i, j in combinations(range(n_classes), 2):
    tprs_ovo[(i, j)] = []
    aucs_ovo[(i, j)] = []

clf_min.fit(X, Y)
feature_names = clf_min.get_feature_names_out()
features_df = pd.DataFrame(index=feature_names)
    
dfOut_skf_all = pd.DataFrame()

# Create StratifiedKFold object
skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=123456)


for ii, (train_index, test_index) in enumerate(skf.split(X, Y, groups)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    clf.fit(X_train, Y_train.values.ravel())
    Y_pred = clf.predict(X_test)
    Y_pred_prob = clf.predict_proba(X_test)

    Y_train_pred = clf.predict(X_train)
    Y_train_proba = clf.predict_proba(X_train)

    # average the probabilities of the 3 classes
    ytpa = np.mean(Y_train_proba, axis=0)    
    
    # set index of Y_pred to be the same as Y_test
    Y_pred_s = pd.Series(Y_pred, index=Y_test.index)
    
    # Save the true and predicted labels
    Y_test_all.extend(Y_test.values)
    Y_pred_all.extend(Y_pred)
    Y_pred_prob_all.extend(Y_pred_prob)
    
    # make a df with y_test and y_pred
    dfOut_skf = pd.concat([Y_test, Y_pred_s], axis=1)
    dfOut_skf.columns = ['manual_h_score', 'predicted_h_score']
    
    # concatenate the dfOut_skf to dfOut_skf_all
    dfOut_skf_all = pd.concat([dfOut_skf_all, dfOut_skf])
    
    # After the model has been fit:
    feature_importances = clf.named_steps['classifier'].feature_importances_
    feature_names = clf[:-1].get_feature_names_out()

    if ii == 0:
        shap_values_all = np.zeros((X.shape[0], len(feature_names), n_classes))
        observations_all = np.zeros((X.shape[0], len(feature_names)))

    # Create a dictionary for holding feature importances
    importances_dict = dict(zip(feature_names, feature_importances))

    # Update features_df with new importances
    features_df[f'Fold {ii}'] = pd.Series(importances_dict)
    
    for c in range(n_classes):
        fpr_c, tpr_c, _ = roc_curve((Y_test == c).astype(int), Y_pred_prob[:, c])
        roc_auc_c = auc(fpr_c, tpr_c)
        interp_tpr_c = np.interp(mean_fpr_ovr, fpr_c, tpr_c)
        tprs_ovr[c].append(interp_tpr_c)
        aucs_ovr[c].append(roc_auc_c)

    for i, j in combinations(range(n_classes), 2):
        mask = np.logical_or(Y_test == i, Y_test == j)
        fpr_ij, tpr_ij, _ = roc_curve((Y_test[mask['manual_h_score']] == i).astype(int), Y_pred_prob[mask['manual_h_score'].ravel(), i])
        roc_auc_ij = auc(fpr_ij, tpr_ij)
        interp_tpr_ij = np.interp(mean_fpr_ovo, fpr_ij, tpr_ij)
        tprs_ovo[(i, j)].append(interp_tpr_ij)
        aucs_ovo[(i, j)].append(roc_auc_ij)
    
    #apply the preprocessing to x_test and feature selection to get shap values
    observations = clf['scaler'].transform(X_test)
    observations_FS = clf['feature_selection'].transform(observations)
    observations_train_FS = clf['feature_selection'].transform(clf['scaler'].transform(X_train))
    
    # Initialize a SHAP explainer with the XGBoost model
    explainer = shap.TreeExplainer(clf.named_steps['classifier'])
    
    # apply softmax transform to explainer.expected_value
    exp_val_prob = np.exp(explainer.expected_value) / np.sum(np.exp(explainer.expected_value))
    
    # observations_FS = observations
    shap_values = explainer.shap_values(observations_FS)

    # concatenate shap values
    shap_values_all[test_index] = shap_values
    observations_all[test_index] = observations_FS

    ######## SHAP DECISION PLOT  ########
    
    # remove "num__original_" from the feature names
    feature_names = [name.replace('original_', '') for name in feature_names]
    # add '_CSA' to the end of the feature names
    feature_names = [name + '_CSA' for name in feature_names]
    # if feature name finishes with "_not_CSA", change it to '_LOW'
    feature_names = [name.replace('_not_CSA', '_LOW') for name in feature_names]
    # change underscores to spaces
    feature_names = [name.replace('_', ' ') for name in feature_names]
    
    # create reprucible seed based on the fold number
    seed = 123456 + ii
    link_f='identity'
    
    generate_shap_plots_for_class(dfOut_skf, X_test, dataDir, result_dir, explainer, feature_names, observations_FS, ii, 1, seed=seed, link_f=link_f)
    generate_shap_plots_for_class(dfOut_skf, X_test, dataDir, result_dir, explainer, feature_names, observations_FS, ii, 2, seed=seed, link_f=link_f)
    generate_shap_plots_for_class(dfOut_skf, X_test, dataDir, result_dir, explainer, feature_names, observations_FS, ii, 3, seed=seed, link_f=link_f)
    
    ######## SHAP DECISION PLOT :: 3 CLASS MERGE  ########
    
    # create a list of the three filenames
    filenames = [f'CorrectClass1_{ii}.png', f'CorrectClass2_{ii}.png', f'CorrectClass3_{ii}.png']
    
    # open the three images
    if link_f == 'logit':
        images = [Image.open(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', filename)) for filename in filenames]
    else:
        images = [Image.open(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', filename)) for filename in filenames]
        
    # get the width and height of the images
    widths, heights = zip(*(i.size for i in images))
    
    # create a new image with the same width and the sum of the heights
    total_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new('RGB', (total_width, total_height))
    
    # paste the three images into the new image
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
        
    # save the new image
    if link_f == 'logit':
        new_im.save(os.path.join(result_dir, 'SHAP', 'CSA', 'logit', f'CorrectClass_{ii}.png'))
    else:
        new_im.save(os.path.join(result_dir, 'SHAP', 'CSA', 'odds', f'CorrectClass_{ii}_odds.png'))

##############
######### FEATURE IMPORTANCE
############## 

# # After completing all folds, calculate average importance and count
# features_df['Average Importance'] = features_df.iloc[:,:10].mean(axis=1)
# features_df['Count'] = (features_df.iloc[:,:10] != 0).sum(axis=1)
# features_df = features_df.sort_values(by=['Count', 'Average Importance'], ascending=False)

# import seaborn as sns
# # Get the 10 most important features
# features_df = features_df[~features_df.index.str.contains('cat')]

# top_features = features_df.loc[features_df['Count']>3,:].index.tolist()
# top_features_avg = features_df['Average Importance'].loc[features_df['Count']>3,].abs()

# # Sort features based on absolute importance
# top_features_avg = top_features_avg.sort_values(ascending=False)

# # remove "num__original_" from the feature names
# top_features_avg.index = top_features_avg.index.str.replace('num__original_', '')

# # add '_CSA' to the end of the feature names
# top_features_avg.index = top_features_avg.index + '_CSA'

# # if feature name finishes with "_not_CSA", change it to '_LOW'
# top_features_avg.index = top_features_avg.index.str.replace('_not_CSA', '_LOW')

# # change underscores to spaces
# top_features_avg.index = top_features_avg.index.str.replace('_', ' ')

# fig = plt.figure(figsize=(10, 6))
# plt.barh(top_features_avg.index[:20], top_features_avg[:20])
# plt.xlabel('Absolute Importance')
# plt.ylabel('Feature')
# plt.title('Top 20 Important Features')
# plt.gca().invert_yaxis()  # reverse the order of features
# # Adjust layout parameters
# plt.tight_layout()
# plt.savefig(os.path.join(result_dir, 'Top20Features.png'), bbox_inches='tight')
# plt.savefig(os.path.join(result_dir, 'Top20Features.svg'), bbox_inches='tight')
# plt.show()

##############
######### SHAP PLOTS
############## 

# create a shap summary plot for each class
for class_index in range(n_classes):
    # Extract SHAP values for the specific class
    class_shap_values = shap_values_all[...,class_index]

    # Create a SHAP summary plot for this class
    shap.summary_plot(class_shap_values, observations_all, feature_names=feature_names, show=False)  # show=False prevents the plot from rendering immediately
    
    # write the plot to a file
    plt.tight_layout()

    plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', f'ShapSummaryPlot_{class_index}.png'), bbox_inches='tight')
    # clear the plot
    plt.show()
    plt.clf()
    
    # Create a SHAP beeswarm plot for this class
    class_shap_explanation = Explanation(values=class_shap_values, 
                                        base_values=explainer.expected_value[class_index], 
                                        data=observations_all, 
                                        feature_names=feature_names)
    
    shap.plots.beeswarm(class_shap_explanation, show=False)  # show=False prevents the plot from rendering immediately
    plt.tight_layout()

    # write the plot to a file
    plt.savefig(os.path.join(result_dir, 'SHAP', 'CSA', f'ShapBeeswarmPlot_{class_index}.png'), bbox_inches='tight')
    # clear the plot
    # plt.show()
    plt.clf()

    
##############
######### CONFUSION MATRIX
############## 

# Compute confusion matrix
Y_test_all = np.array(Y_test_all)
Y_pred_all = np.array(Y_pred_all)
Y_pred_prob_all = np.array(Y_pred_prob_all)

cm = confusion_matrix(Y_test_all, Y_pred_all)
print("Confusion matrix:")
print(cm)

CLASSES = [0,1,2]

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import combinations

cm_all = confusion_matrix(Y_test_all, Y_pred_all)
cm_all_norm = confusion_matrix(Y_test_all, Y_pred_all, normalize='true')

# Set the DPI for better resolution
plt.rcParams['figure.dpi'] = 300

# Create subplots with a shared y-axis
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot the first confusion matrix
ax1 = axes[0]
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_all,
                               display_labels=CLASSES)
disp1.plot(ax=ax1, cmap='Blues')  # Use a colormap that provides good contrast
ax1.set_title('Confusion matrix merging 10 fold', fontsize=12)

# Plot the second confusion matrix
ax2 = axes[1]
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_all_norm,
                               display_labels=CLASSES)
disp2.plot(ax=ax2, cmap='Blues')  # Use a colormap that provides good contrast
ax2.set_title('Confusion matrix merging 10 fold (normalized)', fontsize=12)

# Set labels and titles
plt.suptitle('Comparison of Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'CM', 'ConfusionMatrix_CSAonly.png'), bbox_inches='tight')
plt.savefig(os.path.join(result_dir, 'CM', 'ConfusionMatrix_CSAonly.svg'), bbox_inches='tight')

# get classification report for the total dataset   
from sklearn.metrics import classification_report
print(classification_report(Y_test_all, Y_pred_all))

# compute Cohen's Kappa
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(Y_test_all, Y_pred_all)
print(f"Cohen's Kappa: {kappa}")


##############
######### AUCs
############## 

for i in range(n_classes):
    mean_tpr_ovr[i] = np.mean(tprs_ovr[i], axis=0)
    mean_tpr_ovr[i][-1] = 1.0
    mean_auc_ovr[i] = auc(mean_fpr_ovr, mean_tpr_ovr[i])
    std_auc_ovr[i] = np.std(aucs_ovr[i])

for i, j in combinations(range(n_classes), 2):
    mean_tpr_ovo[(i, j)] = np.mean(tprs_ovo[(i, j)], axis=0)
    mean_tpr_ovo[(i, j)][-1] = 1.0
    mean_auc_ovo[(i, j)] = auc(mean_fpr_ovo, mean_tpr_ovo[(i, j)])
    std_auc_ovo[(i, j)] = np.std(aucs_ovo[(i, j)])
    
# Now you can plot the average ROC curves with uncertainty bounds:
fig, ax = plt.subplots()
for i in range(n_classes):
    ax.plot(mean_fpr_ovr, mean_tpr_ovr[i], label='Avg ROC curve of class {0} (area = {1:0.2f})'.format(i, mean_auc_ovr[i]))
    tprs_upper = np.minimum(mean_tpr_ovr[i] + std_auc_ovr[i], 1)
    tprs_lower = np.maximum(mean_tpr_ovr[i] - std_auc_ovr[i], 0)
    ax.fill_between(mean_fpr_ovr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='_nolegend_')
    
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (OvR)')
ax.legend(loc="lower right")
plt.savefig(os.path.join(result_dir,'CM', 'RocOVR_CSAonly.png'), bbox_inches='tight')
plt.savefig(os.path.join(result_dir,'CM', 'RocOVR_CSAonly.svg'), bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
for i, j in combinations(range(n_classes), 2):
    ax.plot(mean_fpr_ovo, mean_tpr_ovo[(i, j)], label='Avg ROC curve of classes {0} vs {1} (area = {2:0.2f})'.format(i, j, mean_auc_ovo[(i, j)]))
    tprs_upper = np.minimum(mean_tpr_ovo[(i, j)] + std_auc_ovo[(i, j)], 1)
    tprs_lower = np.maximum(mean_tpr_ovo[(i, j)] - std_auc_ovo[(i, j)], 0)
    ax.fill_between(mean_fpr_ovo, tprs_lower, tprs_upper, color='grey', alpha=.2, label='_nolegend_')
    
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (OvO)')
ax.legend(loc="lower right")
plt.savefig(os.path.join(result_dir,'CM', 'RocOVO_CSAonly.png'), bbox_inches='tight')
plt.savefig(os.path.join(result_dir,'CM', 'RocOVO_CSAonly.svg'), bbox_inches='tight')
plt.show()

# create a dataframe with the predicted and manual h scores
dfOut_skf_all.head()

# add 1 to the predicted_h_score and manual_h_score to have the classes starting from 1, convert to int before
dfOut_skf_all['predicted_h_score'] = dfOut_skf_all['predicted_h_score'].astype(int) + 1
dfOut_skf_all['manual_h_score'] = dfOut_skf_all['manual_h_score'].astype(int) + 1

# put column EI of df_H_Z_1 in dfOut_skf_all following the index
dfOut_skf_all['EI'] = df_H_Z_1['EI']

# create column Muscle in dfOut_skf_all using the second item in the index
dfOut_skf_all['Muscle'] = dfOut_skf_all.index.get_level_values(1)
# convert the values of Muscle to muscle names using code_to_class_original
dfOut_skf_all['Muscle'] = dfOut_skf_all['Muscle'].map(code_to_class_original)

##############
######### BOXPLOTS HECKMATT AND ZSCORE
##############

# Setting the style and context for the plot
sns.set_style("whitegrid")
sns.set_context("talk")

# Initialize the matplotlib figure
plt.figure(figsize=(14, 8))

# Create the boxplot, adjust dodge parameter if needed
boxplot = sns.boxplot(x='Muscle', y='EI', hue='manual_h_score', data=dfOut_skf_all, palette='Greys', dodge=True)

# Final touches on the plot
plt.title('Relationship between EI and H grouped by Muscle')
plt.ylabel('EI Value')
plt.xlabel('Muscle')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels by 45 degrees and align them to the right
plt.legend(title='H category', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.ylim(-7, 13)
plt.tight_layout()
# save the plot
plt.savefig(os.path.join(result_dir, 'HECKMATT', 'BoxplotEIvsH_CSAonly.png'), bbox_inches='tight')
plt.savefig(os.path.join(result_dir, 'HECKMATT', 'BoxplotEIvsH_CSAonly.svg'), bbox_inches='tight')
plt.show()

# Initialize the matplotlib figure
plt.figure(figsize=(14, 8))

# Create the boxplot, adjust dodge parameter if needed
boxplot = sns.boxplot(x='Muscle', y='EI', hue='predicted_h_score', data=dfOut_skf_all, palette='Greys', dodge=True)

# Final touches on the plot
plt.title('Relationship between EI and H grouped by Muscle')
plt.ylabel('EI Value')
plt.xlabel('Muscle')
plt.legend(title='H category', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.ylim(-7, 13)
plt.tight_layout()
# save the plot
plt.savefig(os.path.join(result_dir, 'HECKMATT', 'BoxplotEIvsH_predicted_CSAonly.png'), bbox_inches='tight')
plt.savefig(os.path.join(result_dir, 'HECKMATT', 'BoxplotEIvsH_predicted_CSAonly.svg'), bbox_inches='tight')
plt.show()

# calculate Spearman rank correlation between 'EI' and 'H' for each muscle  
correlations_manual = dfOut_skf_all.groupby('Muscle').apply(lambda x: x[['EI', 'manual_h_score']].corr(method='spearman').iloc[0, 1])
print('Spearman rank correlation between EI and manual H for each muscle')
print(correlations_manual)

# calculate Spearman rank correlation between 'EI' and 'H' for each muscle  
correlations = dfOut_skf_all.groupby('Muscle').apply(lambda x: x[['EI', 'predicted_h_score']].corr(method='spearman').iloc[0, 1])
print('Spearman rank correlation between EI and predicted H for each muscle')
print(correlations)

# make a dataframe with the correlation values both manual and predicted for each muscle
correlations_df = pd.DataFrame({'Manual H': correlations_manual, 'Predicted H': correlations})
correlations_df = correlations_df.reset_index()

# save the dataframe to a csv file
correlations_df.to_csv(os.path.join(excel_dir,'CorrelationEIvsH_CSAonly.csv'), index=False)

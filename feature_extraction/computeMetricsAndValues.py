import numpy as np
from PIL import Image
from skimage import morphology
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
import torch
from torchvision import models, transforms

from radiomics import featureextractor
import logging

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


import json
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# avoid plotting plt figures to screen
plt.ioff()

import numpy as np

### DEFINE VISUALIZATION FUNCTIONS
            
def display_confusion_matrix_and_scores(y_true, y_pred, labels, fold='total', experiment='knet_swin_binary'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:")
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {fold}')
    plt.xticks()
    plt.savefig(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/CONFUSION_MATRIX/{experiment}_confusion_matrix_{fold}.png', dpi=300, bbox_inches='tight')
    # plt.show()

    normalized_cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps)
    print("Normalized Confusion Matrix:")
    plt.figure(figsize=(16, 14))
    sns.heatmap(normalized_cm, annot=True, fmt='.1%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Normalized Confusion Matrix for {fold}')
    plt.xticks()
    plt.savefig(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/CONFUSION_MATRIX/{experiment}_normalized_confusion_matrix_{fold}.png', dpi=300, bbox_inches='tight')
    # plt.show()

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print("Classification Report:")
    print(report)
    
    # save report to xlsx file
    report_df = pd.DataFrame(classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)).transpose()
    report_df.to_excel(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/{experiment}_classification_report_{fold}.xlsx', sheet_name=f'{fold}')


##########
###### LOAD HECKMATT FROM JSON FILES AND CREATE EXCEL FOR DOCS
##########  

dataDir = '/home/francesco/Desktop/POLI/RADBOUD/'
filename = os.path.join(dataDir, 'DATA', 'TABULAR', 'heckMapPlusCharacteristics.xlsx')
HeckMap = pd.read_excel(filename)
HeckMap = HeckMap.iloc[5:,:]

# filename = "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/segmentation_summary_knet_swin_binary_GTguide_pred.json"
# filename = "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/segmentation_summary_knet_swin_mod_GTguide_pred.json"
# filename = "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/segmentation_summary_knet_swin_mod_pred.json"
filename = "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/segmentation_summary_knet_swin_mod_muscle_specific.xlsx"

output_file = "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/AAA_resultSegmentation_knet_mod_muscle_specific.xlsx"

experiment = 'knet_swin_muscle_specific'

data = json.load(open(filename, "r"))
df = pd.DataFrame.from_dict(data)

del data
import gc
gc.collect()

# remove rows with slice number > 90
df['Slice'] = df['File'].apply(lambda x: x.split('_')[-1].split('.')[0])
df = df[df['Slice'].astype(int) <= 90]

root = '/home/francesco/Desktop/POLI/RADBOUD/'

# take all the images in the 5 test sets and count them
test_files = []
for fold in range(5):
    fold_dir = os.path.join(root, 'DATA','DEVELOPMENT', 'FSHD_v3_' + 'f' + str(fold))
    img_dir = os.path.join(fold_dir, 'images')
    test_dir = os.path.join(img_dir, 'testing')
    test_files += os.listdir(test_dir)

# search for the missing filenames beween df['File'] and filenames
missing_filenames = [x for x in test_files if x not in df['File'].values]

# save list to txt file
with open('/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/missing_filenames.txt', 'w') as f:
    for item in missing_filenames:
        f.write("%s\n" % item)
    

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

# Creating the new 'muscle_code' column
df['muscle_code'] = df['class_gt'].map(class_to_code)

# Create a new column 'original_firstorder_Mean'
df['gt_mean'] = df['features_img_gt'].apply(
    lambda x: x.get('original_firstorder_Mean') if x != 'mask not found' else np.nan)

# Create a new column 'original_firstorder_Mean'
df['pred_mean'] = df['features_img_pred'].apply(
    lambda x: x.get('original_firstorder_Mean') if x != 'mask not found' else np.nan)

# Merging the two dataframes
HeckMap['Code'] = HeckMap['Code'].apply(lambda x: str(int(float(x))).zfill(5) if pd.notnull(x) and x != '' else '')

HeckMap['Code'] = HeckMap['Code'].astype(str)
HeckMap['Sex'] = HeckMap['Sex'].astype(str)
HeckMap['FSHD_age'] = HeckMap['FSHD_age'].astype(str)
HeckMap['FSHD_BMI'] = HeckMap['FSHD_BMI'].astype(str)

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

df1_head = df1.head(100)

# Polish the result and retain only useful columns
df2 = df1.loc[:,['Fold','File','subject','muscle','side','age','sex','bmi',
                 'class_gt','class_pred','gt_mean','pred_mean','manual_h_score','iou']]

# Convert 'class_gt' and 'class_pred' to numeric, converting errors to NaN
df2['gt_mean'] = pd.to_numeric(df2['gt_mean'], errors='coerce')
df2['pred_mean'] = pd.to_numeric(df2['pred_mean'], errors='coerce')

# Now you can proceed with the grouping and averaging
aggregations = {col: 'mean' if df2[col].dtype.kind in 'biufc' else 'first' for col in df2.columns.difference(['subject', 'muscle','side'])}
df2_grouped = df2.groupby(['subject', 'muscle','side']).agg(aggregations).reset_index()

# Write each dataframe to a different worksheet.
df2.to_excel(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/{experiment}_file_for_zscore.xlsx',sheet_name='Original Data')
df2_grouped.to_excel(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/{experiment}_file_for_zscore_grouped.xlsx',sheet_name='Grouped Data')

a = df2.head()

from scipy import stats
classes = df['class_gt'].unique()
palette = sns.color_palette('hsv', len(classes))

df_plot = df2_grouped.loc[:,['subject','muscle','side','age','sex','bmi','class_gt','gt_mean','pred_mean','manual_h_score','iou']]
df_plot['mean_diff'] = df_plot['gt_mean']-df_plot['pred_mean']
df_plot['mean_diff_perc'] = (df_plot['gt_mean']-df_plot['pred_mean'])/df_plot['gt_mean']
df_plot.to_excel(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/{experiment}_file_for_plots.xlsx',sheet_name='Grouped Data')

# Remove rows with NaN or infinite values in 'gt_mean' or 'pred_mean'
data = df_plot[~(df_plot[['gt_mean', 'pred_mean']].isnull().any(axis=1) | 
              df_plot[['gt_mean', 'pred_mean']].applymap(np.isinf).any(axis=1))]


# # Create a scatter plot for each class
# for class_name in classes:
#     if not 'background'==class_name:
#         # Filter the data for the current class
#         class_data = data[data['class_gt'] == class_name]
        
#         # Remove rows with NaN or infinite values in 'gt_mean' or 'pred_mean'
#         class_data = class_data[~(class_data[['gt_mean', 'pred_mean']].isnull().any(axis=1) | 
#                                   class_data[['gt_mean', 'pred_mean']].applymap(np.isinf).any(axis=1))]
        
#         # Create a figure with two subplots
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

#         # First subplot: Correlation between gt_mean and pred_mean
#         ax1 = axes[0]
#         sns.scatterplot(x='gt_mean', y='pred_mean', color='blue', data=class_data, ax=ax1)
#         sns.regplot(x='gt_mean', y='pred_mean', scatter=False, color='red', data=class_data, ax=ax1)

#         # Compute the correlation coefficient and the p-value
#         corr_coef, p_value = stats.pearsonr(class_data['gt_mean'], class_data['pred_mean'])

#         # Add the correlation coefficient, p-value, and regression line equation to the plot
#         ax1.text(min(class_data['gt_mean']), max(class_data['pred_mean']),
#                   f'r={corr_coef:.2f}\np={p_value:.2e}',
#                   va='top', fontsize=12)

#         ax1.set_title(f'Correlation between gt_mean and pred_mean ({class_name})', fontsize=12)
#         ax1.set_xlabel('gt_mean', fontsize=12)
#         ax1.set_ylabel('pred_mean', fontsize=12)

#         # Second subplot: Correlation between iou and gt_mean
#         ax2 = axes[1]
#         sns.scatterplot(x='iou', y='gt_mean', color='blue', data=class_data, ax=ax2)
#         sns.regplot(x='iou', y='gt_mean', scatter=False, color='red', data=class_data, ax=ax2)

#         # Compute the correlation coefficient and the p-value
#         corr_coef, p_value = stats.pearsonr(class_data['iou'], class_data['gt_mean'])

#         # Add the correlation coefficient, p-value, and regression line equation to the plot
#         ax2.text(min(class_data['iou']), max(class_data['gt_mean']),
#                   f'r={corr_coef:.2f}\np={p_value:.2e}',
#                   va='top')

#         ax2.set_title(f'Correlation between iou and gt_mean ({class_name})', fontsize=12)
#         ax2.set_xlabel('iou', fontsize=12)
#         ax2.set_ylabel('gt_mean', fontsize=12)

#         # Adjust the spacing between subplots
#         plt.tight_layout()

#         # Display the plot
#         plt.show()
#         fig.savefig(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/CORRELATION/knet_swin_binary_Correlation_plots_{class_name}.png')

##########
###### PLOT CONFUSION MATRIX FOR MUSCLE CLASSIFICATION
##########

# Assuming 'df' is your DataFrame and 'class_gt' is one of its columns.
class_counts = df['class_gt'].value_counts()

# Define a color palette
# palette = sns.cubehelix_palette(n_colors=len(class_counts.index), start=2, rot=0, dark=0.2, light=0.8, reverse=True)
palette = sns.color_palette("Blues_r", n_colors=len(class_counts.index))

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(y=class_counts.index, x=class_counts.values, palette=palette)
plt.xlabel('Count')
plt.ylabel('muscle type')
plt.title('Muscle type distribution')
plt.xticks(rotation=90)

# Adjust layout to fit labels
plt.tight_layout()
plt.savefig(f"/home/francesco/Desktop/POLI/RADBOUD/RESULTS/{experiment}_class_distribution.png", dpi=300)
# plt.show()

# add 'background' to classes
classes = np.append(classes, 'background')

# Make a copy of the DataFrame
df_copy = df1.copy()

# Retain only the row where manual_h_score is not NaN
# df_copy = df_copy[~df_copy['manual_h_score'].isnull()]

temp_classes = classes.copy()
# retain only the classes background and the ones with at least 10 samples
for cl in classes:
    if cl != 'background' and len(df_copy[df_copy['class_gt'] == cl]) < 10:
        temp_classes = temp_classes[temp_classes != cl]

print("=== Confusion matrix and scores for the total dataset ===")
display_confusion_matrix_and_scores(df_copy['class_gt'], df_copy['class_pred'], temp_classes)

cm = confusion_matrix(df_copy['class_gt'], df_copy['class_pred'], labels=temp_classes)
normalized_cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps)

# Unique fold values
folds = df_copy['Fold'].unique()

for fold in folds:
    print(f"=== Confusion matrix and scores for Fold: {fold} ===")
    subset_df = df_copy[df_copy['Fold'] == fold]
    display_confusion_matrix_and_scores(subset_df['class_gt'], subset_df['class_pred'], temp_classes, fold)
    
# Look into the errors to check dataset
errors_df = df_copy[df_copy['class_gt'] != df_copy['class_pred']].copy()
errors_df.to_excel('clss_err_to_check.xlsx')
    
##########
###### PLOT BOXPLOTS AND COMPUTE SEGMENTATION METRICS
##########

# Make a copy of the DataFrame
df_copy = df1.copy()

# Mask where class_gt and class_pred are not equal
mask = df_copy['class_gt'] != df_copy['class_pred']

# Retain only the row where manual_h_score is not NaN
# df_copy = df_copy[~df_copy['manual_h_score'].isnull()]

# Calculate mean and standard deviation grouped by 'class_gt' and 'Fold'
mean = df_copy.groupby(['class_gt', 'Fold'])[['iou', 'prec', 'rec']].mean()
std_dev = df_copy.groupby(['class_gt', 'Fold'])[['iou', 'prec', 'rec']].std()

cm = confusion_matrix(df['class_gt'], df['class_pred'], labels=classes)
mean = df_copy.groupby(['class_gt'])[['iou', 'prec', 'rec']].mean()
std_dev = df_copy.groupby(['class_gt'])[['iou', 'prec', 'rec']].std()

# Calculate the percentage of correct classified for each class
correct_percentage = pd.DataFrame({'Correct Percentage': (cm.diagonal() / cm.sum(axis=1)) * 100},
                                  index=classes).round(2)
# Calculate the percentage of entries with 'iou' lower than 0.2 for each class
low_iou_percentage = pd.DataFrame({'Low IoU Percentage': (df_copy['iou'] < 0.2).groupby(df_copy['class_gt']).mean() * 100},
                                  index=classes).round(2)

# Concatenate the correct percentage, mean, and standard deviation DataFrames
summary_table = pd.concat([correct_percentage, low_iou_percentage, mean, std_dev], axis=1)

# Format the values in the summary table to two decimal places
summary_table = summary_table.round(2)

# Save the summary table to an Excel file
summary_table.to_excel('paper_table.xlsx', index_label='class_gt')

# Create a DataFrame with mean and standard deviation in the required format
df_result = mean.copy()
for col in df_result.columns:
    df_result[col] = df_result[col].map('{:.2f}'.format) + ' +/- ' + std_dev[col].map('{:.2f}'.format)

# Save the result DataFrame to an Excel file
df_result.to_excel(output_file)

# Create boxplots
for column in ['iou', 'prec', 'rec']:
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='class_gt', y=column, hue='Fold', data=df_copy)
    plt.title(f'Boxplot of {column} grouped by class_gt and Fold')
    plt.xticks(rotation=90)
    plt.savefig(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/BOXPLOT/{experiment}_boxplot_{column}_by_fold.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create a single boxplot for the whole dataset
for column in ['iou', 'prec', 'rec']:
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='class_gt', y=column, data=df_copy)
    plt.title(f'Boxplot of {column} grouped by class_gt')
    plt.xticks(rotation=90)
    plt.savefig(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/BOXPLOT/{experiment}_boxplot_{column}_total.png', dpi=300, bbox_inches='tight')
    plt.show()

##########
# Add the calculation of the mean intersection over union across all the dataset groupin the results by folds

##########
###### COMPARE TEXTURE FEATURES FROM GT AND PRED
##########    

# # Convert dictionaries in 'features_img_gt' and 'features_img_pred' columns to DataFrames
# df_gt = pd.json_normalize(df['features_img_gt'])
# df_pred = pd.json_normalize(df['features_img_pred'])

# # Replace 'mask not found' with NaN
# df_gt.replace('mask not found', np.nan, inplace=True)
# df_pred.replace('mask not found', np.nan, inplace=True)

# # Convert columns to numeric
# df_gt = df_gt.apply(pd.to_numeric, errors='coerce')
# df_pred = df_pred.apply(pd.to_numeric, errors='coerce')

# nan_rows = df_pred[df_pred.isna().any(axis=1)]

# df_gt = df_gt[~df_gt.index.isin(nan_rows.index)]
# df_pred = df_pred[~df_pred.index.isin(nan_rows.index)]

# nan_rows = df_gt[df_gt.isna().any(axis=1)]

# df_gt = df_gt[~df_gt.index.isin(nan_rows.index)]
# df_pred = df_pred[~df_pred.index.isin(nan_rows.index)]

# # Store the results
# correlations = {}
# p_values = {}

# # Loop over each feature
# for column in df_gt.columns:
#     # Check if column exists in both DataFrames
#     if column in df_pred:
#         # Drop rows with NaN values in either DataFrame
#         temp_gt = df_gt[column]
#         temp_pred = df_pred[column]
#         common = temp_gt.index.intersection(temp_pred.index)
        
#         # Calculate correlation coefficient
#         corr, _ = pearsonr(temp_gt.loc[common], temp_pred.loc[common])
#         correlations[column] = corr

#         # Perform two-sample t-test
#         t_stat, p_value = ttest_ind(temp_gt.loc[common], temp_pred.loc[common])
#         p_values[column] = p_value

# # Convert results to DataFrames
# df_correlations = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation Coefficient'])
# df_p_values = pd.DataFrame.from_dict(p_values, orient='index', columns=['P-value'])

# print("Correlation Coefficients:")
# print(df_correlations)
# print("\nP-values:")
# print(df_p_values)

# # how many features with correlation above 0.8?
# print(len(df_correlations[df_correlations['Correlation Coefficient'] > 0.8]))
# print(len(df_correlations[df_correlations['Correlation Coefficient'] > 0.9]))
# print(len(df_correlations[df_correlations['Correlation Coefficient'] > 0.95]))

##########
###### MAKE t-SNE PLOTS TO VISUALIZE MUSCLE CLASSES USING FEATURES
##########    

# # Convert dictionaries to DataFrames
# df_gt = pd.json_normalize(df['features_img_gt'])

# # Replace 'mask not found' with NaN
# df_gt.replace('mask not found', np.nan, inplace=True)

# # Convert columns to numeric
# df_gt = df_gt.apply(pd.to_numeric, errors='coerce')

# eb_data = df_gt.values
# mask = ~np.isnan(eb_data).any(axis=1)
# eb_data = eb_data[mask]

# types = df['class_gt'][mask]

# # standardize the numerical features
# scaler = StandardScaler()
# eb_data_scaled = scaler.fit_transform(eb_data)

# # Apply PCA 
# pca = PCA(n_components=2)
# eb_data_pca = pca.fit_transform(eb_data_scaled)

# # Plot PCA results adding a color for each class
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x=eb_data_pca[:,0], y=eb_data_pca[:,1],
#     hue=types, # The points will be colored based on the 'types' variable
#     palette=sns.color_palette("hsv", len(df['class_gt'].unique())), # Different colors for each type
#     legend="full",
#     alpha=0.8
# )
# plt.show()

# # Apply t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(eb_data_scaled)

# # Plot t-SNE results adding a color for each class
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x=tsne_results[:,0], y=tsne_results[:,1],
#     hue=types, # The points will be colored based on the 'types' variable
#     palette=sns.color_palette("hsv", len(df['class_gt'].unique())), # Different colors for each type
#     legend="full",
#     alpha=0.8
# )
# plt.show()

# # Applu UMAP
# import umap.umap_ as umap
# reducer = umap.UMAP(random_state=42)
# umap_results = reducer.fit_transform(eb_data_scaled)

# # Plot UMAP results adding a color for each class
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x=umap_results[:,0], y=umap_results[:,1],
#     hue=types, # The points will be colored based on the 'types' variable
#     palette=sns.color_palette("hsv", len(df['class_gt'].unique())), # Different colors for each type
#     legend="full",
#     alpha=0.8
# )
# plt.show()


# # Save the plot as a PNG file with a resolution of 300 dpi
# plt.savefig(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/TSNE/knet_swin_binary_{col}_tsne.png', dpi=300)

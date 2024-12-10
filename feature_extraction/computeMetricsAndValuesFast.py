import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import gc
from matplotlib.colors import ListedColormap  # Import for custom colormap

# Avoid plotting plt figures to screen
plt.ioff()

# Define base directories
BASE_DIR = Path('/home/francesco/Desktop/POLI/RADBOUD/')
DATA_DIR = BASE_DIR / 'DATA'
RESULTS_DIR = BASE_DIR / 'RESULTS'
TABULAR_DIR = DATA_DIR / 'TABULAR'


def display_confusion_matrix_and_scores(y_true, y_pred, labels, fold='total', experiment='knet_swin_binary', output_dir=None):
    """Displays and saves confusion matrices and classification reports."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:")
    plt.figure(figsize=(16, 14))
    cmap = ListedColormap(sns.color_palette("tab10", n_colors=256).as_hex())  # Using 'tab10' palette
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {fold}')
    plt.xticks(rotation=90)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'{experiment}_confusion_matrix_{fold}.png', dpi=300, bbox_inches='tight')
    plt.close()

    normalized_cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps)
    print("Normalized Confusion Matrix:")
    plt.figure(figsize=(16, 14))
    sns.heatmap(normalized_cm, annot=True, fmt='.1%', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Normalized Confusion Matrix for {fold}')
    plt.xticks(rotation=90)
    if output_dir:
        plt.savefig(output_dir / f'{experiment}_normalized_confusion_matrix_{fold}.png', dpi=300, bbox_inches='tight')
    plt.close()

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    excel_dir = RESULTS_DIR / 'EXCEL'
    excel_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_excel(excel_dir / f'{experiment}_classification_report_{fold}.xlsx', sheet_name=f'{fold}')
    print("Classification Report saved to Excel.")


def load_data(experiment):
    """Loads HeckMap and segmentation summary data."""
    # Load HeckMap data
    heckmap_file = TABULAR_DIR / 'heckMapPlusCharacteristics.xlsx'
    heckmap = pd.read_excel(heckmap_file)
    heckmap = heckmap.iloc[5:, :]

    # Load segmentation summary data
    segmentation_summary_file = RESULTS_DIR / 'EXCEL' / f'segmentation_summary_{experiment}.json'
    with open(segmentation_summary_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data)
    del data
    gc.collect()
    return heckmap, df


def process_data(df, heckmap):
    """Processes data by merging and cleaning."""
    df['Slice'] = df['File'].apply(lambda x: x.split('_')[-1].split('.')[0])
    df = df[df['Slice'].astype(int) <= 90]

    # Map 'class_gt' to 'muscle_code'
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
    df['muscle_code'] = df['class_gt'].map(class_to_code)

    # Extract 'gt_mean' and 'pred_mean'
    df['gt_mean'] = df['features_img_gt'].apply(
        lambda x: x.get('original_firstorder_Mean') if isinstance(x, dict) else np.nan)
    df['pred_mean'] = df['features_img_pred'].apply(
        lambda x: x.get('original_firstorder_Mean') if isinstance(x, dict) else np.nan)

    # Prepare HeckMap data
    heckmap['Code'] = heckmap['Code'].apply(lambda x: str(int(float(x))).zfill(5) if pd.notnull(x) and x != '' else '')
    heckmap['Code'] = heckmap['Code'].astype(str)
    heckmap['Sex'] = heckmap['Sex'].astype(str)
    heckmap['FSHD_age'] = heckmap['FSHD_age'].astype(str)
    heckmap['FSHD_BMI'] = heckmap['FSHD_BMI'].astype(str)

    # Merge df and heckmap
    df_merged = pd.merge(df, heckmap[['Code', 'Sex', 'FSHD_age', 'FSHD_BMI']], left_on='subject', right_on='Code', how='left')
    df_merged = df_merged.drop('Code', axis=1)
    df_merged.rename(columns={'Sex': 'sex', 'FSHD_age': 'age', 'FSHD_BMI': 'bmi'}, inplace=True)

    # Create 'muscle_side' column
    df_merged['muscle_side'] = df_merged['muscle_code'].astype(str) + '_' + df_merged['side'].astype(str)

    # Create 'manual_h_score' column
    def get_manual_h_score(row):
        column_name = row['muscle_side']
        subject_name = row['subject']
        if column_name in heckmap.columns and subject_name in heckmap['Code'].values:
            find_idx = heckmap.index[heckmap['Code'] == subject_name][0]
            return heckmap.at[find_idx, column_name]
        else:
            return np.nan
    df_merged['manual_h_score'] = df_merged.apply(get_manual_h_score, axis=1)
    return df_merged


def compute_mean_iou_per_fold(df):
    """Calculates mean IoU grouped by folds."""
    mean_iou = df.groupby('Fold')['iou'].mean().reset_index()
    std_iou = df.groupby('Fold')['iou'].std().reset_index()
    return mean_iou, std_iou


def plot_boxplots(df, experiment):
    """Creates and saves boxplots for segmentation metrics."""
    boxplot_dir = RESULTS_DIR / 'BOXPLOT'
    boxplot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ['iou', 'prec', 'rec']
    for metric in metrics:
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='class_gt', y=metric, hue='Fold', data=df, palette='tab10')  # Using 'tab10' palette
        plt.title(f'Boxplot of {metric} grouped by class_gt and Fold')
        plt.xticks(rotation=90)
        plt.savefig(boxplot_dir / f'{experiment}_boxplot_{metric}_by_fold.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Create a single boxplot for the whole dataset
    for metric in metrics:
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='class_gt', y=metric, data=df, palette='tab10')  # Using 'tab10' palette
        plt.title(f'Boxplot of {metric} grouped by class_gt')
        plt.xticks(rotation=90)
        plt.savefig(boxplot_dir / f'{experiment}_boxplot_{metric}_total.png', dpi=300, bbox_inches='tight')
        plt.close()


def compute_and_save_summary_tables(df, classes, experiment):
    """Computes summary statistics and saves them to Excel."""
    excel_dir = RESULTS_DIR / 'EXCEL'
    excel_dir.mkdir(parents=True, exist_ok=True)
    output_file = excel_dir / f'{experiment}_summary_tables.xlsx'

    # Calculate mean and std dev grouped by 'class_gt'
    mean_metrics = df.groupby('class_gt')[['iou', 'prec', 'rec']].mean()
    std_metrics = df.groupby('class_gt')[['iou', 'prec', 'rec']].std()

    # Compute confusion matrix
    cm = confusion_matrix(df['class_gt'], df['class_pred'], labels=classes)

    # Calculate the percentage of correct classifications for each class
    correct_percentage = pd.DataFrame({
        'Correct Percentage': (cm.diagonal() / cm.sum(axis=1)) * 100
    }, index=classes).round(2)

    # Calculate the percentage of entries with 'iou' lower than 0.2 for each class
    low_iou_percentage = pd.DataFrame({
        'Low IoU Percentage': df.groupby('class_gt')['iou'].apply(lambda x: (x < 0.2).mean() * 100)
    }, index=classes).round(2)

    # Concatenate the DataFrames
    summary_table = pd.concat([correct_percentage, low_iou_percentage, mean_metrics, std_metrics], axis=1)
    summary_table = summary_table.round(2)

    # Create a DataFrame with mean and std in the format 'mean +/- std'
    metrics_formatted = mean_metrics.copy()
    for col in metrics_formatted.columns:
        metrics_formatted[col] = mean_metrics[col].map('{:.2f}'.format) + ' +/- ' + std_metrics[col].map('{:.2f}'.format)

    # Save to Excel
    with pd.ExcelWriter(output_file) as writer:
        summary_table.to_excel(writer, sheet_name='Summary', index_label='class_gt')
        metrics_formatted.to_excel(writer, sheet_name='Metrics Formatted', index_label='class_gt')
    print(f"Summary tables saved to {output_file}")


def main():
    experiment = 'knet_swin_mod_muscle_specific'

    # Load data
    heckmap, df = load_data(experiment)

    # Process data
    df_processed = process_data(df, heckmap)

    # Compute mean IoU per fold
    mean_iou_by_fold, std_iou_by_fold = compute_mean_iou_per_fold(df_processed)
    print("Mean IoU by Fold:")
    print(mean_iou_by_fold)
    print("Std IoU by Fold:")
    print(std_iou_by_fold)

    # Save mean and std IoU per fold to Excel in the same file
    excel_dir = RESULTS_DIR / 'EXCEL'
    excel_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_dir / f'{experiment}_iou_by_fold.xlsx') as writer:
        mean_iou_by_fold.to_excel(writer, sheet_name='mean_iou', index=False)
        std_iou_by_fold.to_excel(writer, sheet_name='std_iou', index=False)

    # Plot confusion matrices
    classes = [cl for cl in df_processed['class_gt'].unique() if cl != 'background']
    confusion_matrix_dir = RESULTS_DIR / 'CONFUSION_MATRIX'

    # Overall confusion matrix
    display_confusion_matrix_and_scores(
        df_processed['class_gt'], df_processed['class_pred'], classes, 'total', experiment, confusion_matrix_dir)

    # Confusion matrix per fold
    folds = df_processed['Fold'].unique()
    for fold in folds:
        print(f"=== Confusion matrix and scores for Fold: {fold} ===")
        subset_df = df_processed[df_processed['Fold'] == fold]
        display_confusion_matrix_and_scores(
            subset_df['class_gt'], subset_df['class_pred'], classes, fold, experiment, confusion_matrix_dir)

    # Plot boxplots
    plot_boxplots(df_processed, experiment)

    # Compute and save summary tables
    compute_and_save_summary_tables(df_processed, classes, experiment)


if __name__ == "__main__":
    main()

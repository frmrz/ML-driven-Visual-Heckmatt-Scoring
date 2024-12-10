import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# Set plot style
sns.set(style="whitegrid")

# Define the root directory and file paths
root = "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL"
original_path = os.path.join(root, "segmentation_summary_knet_swin_mod_pred.xlsx")
binary_path = os.path.join(root, "segmentation_summary_knet_swin_binary_GTguide_pred.xlsx")
muscle_spec_path = os.path.join(root, "segmentation_summary_knet_swin_mod_muscle_specific.xlsx")

# Load data into DataFrames
df_original = pd.read_excel(original_path)
df_binary = pd.read_excel(binary_path)
df_muscle_spec = pd.read_excel(muscle_spec_path)

# Display the first few rows of the original DataFrame
print("Original DataFrame Preview:")
print(df_original.head())

# Add a 'Method' column to identify the dataset source
df_original['Method'] = 'Original'
df_binary['Method'] = 'Binary'
df_muscle_spec['Method'] = 'Muscle Specific'

# Select the necessary columns
columns_needed = ['iou', 'prec', 'rec', 'File', 'class_gt', 'Method']
df_original = df_original[columns_needed]
df_binary = df_binary[columns_needed]
df_muscle_spec = df_muscle_spec[columns_needed]

# From here edit the code to include the third method in the comparison, the third method will be available only for one class_gt

# Combine the DataFrames for plotting
df_combined = pd.concat([df_original, df_binary, df_muscle_spec], ignore_index=True)

# Initialize a list to collect statistical results
stat_results = []

# List of metrics to analyze
metrics = ['iou', 'prec', 'rec']

# Perform normality tests and statistical tests for each muscle in class_gt
alpha = 0.05  # Significance level

print("\nStatistical Analysis Results:")
for metric in metrics:
    for muscle in df_combined['class_gt'].unique():
        print(f"\nAnalyzing '{metric}' for muscle '{muscle}':")
        
        # Get the methods available for this muscle
        available_methods = df_combined[df_combined['class_gt'] == muscle]['Method'].unique()
        
        if len(available_methods) < 2:
            print("  Not enough methods to perform comparisons.")
            continue
        
        # Generate all pairs of methods to compare
        method_pairs = list(combinations(available_methods, 2))
        
        for method1, method2 in method_pairs:
            print(f"Comparing {method1} vs {method2}:")
            
            # Extract data for each method and muscle
            data1 = df_combined[(df_combined['class_gt'] == muscle) & (df_combined['Method'] == method1)][metric].dropna()
            data2 = df_combined[(df_combined['class_gt'] == muscle) & (df_combined['Method'] == method2)][metric].dropna()
            
            # Check if there is enough data to perform the tests
            if len(data1) < 3 or len(data2) < 3:
                print("  Not enough data to perform statistical tests.")
                continue
            
            # Normality tests using Shapiro-Wilk test
            stat1, p1 = shapiro(data1)
            stat2, p2 = shapiro(data2)
            
            normal1 = p1 > alpha
            normal2 = p2 > alpha
            
            # Choose statistical test based on normality
            if normal1 and normal2:
                # Use independent t-test
                stat_indep, p_value_indep = ttest_ind(data1, data2)
                test_name_indep = "Independent t-test"
            else:
                # Use Mann-Whitney U test
                stat_indep, p_value_indep = mannwhitneyu(data1, data2)
                test_name_indep = "Mann-Whitney U test"
            
            # Interpret the statistical test result
            if p_value_indep > alpha:
                result_indep = "No significant difference"
            else:
                result_indep = "Significant difference"
            
            # Append the independent test results to the list
            stat_results.append({
                'Metric': metric,
                'Muscle': muscle,
                'Methods Compared': f"{method1} vs {method2}",
                'Test': test_name_indep,
                'Statistic': stat_indep,
                'p-value': p_value_indep,
                'Result': result_indep
            })
            
            # Print independent test results
            print(f"  {test_name_indep}: Statistic={stat_indep:.4f}, p-value={p_value_indep:.4f}")
            print(f"  Result: {result_indep} between {method1} and {method2} for muscle '{muscle}'.")
            
            # For paired tests, we need to check if we have paired data
            # Merge the data on 'File' and 'class_gt' for the two methods
            df_method1 = df_combined[(df_combined['class_gt'] == muscle) & (df_combined['Method'] == method1)][['File', metric]]
            df_method2 = df_combined[(df_combined['class_gt'] == muscle) & (df_combined['Method'] == method2)][['File', metric]]
            
            df_paired_methods = pd.merge(df_method1, df_method2, on='File', suffixes=(f'_{method1}', f'_{method2}'))
            
            data_method1 = df_paired_methods[f'{metric}_{method1}']
            data_method2 = df_paired_methods[f'{metric}_{method2}']
            
            if len(data_method1) < 3 or len(data_method2) < 3:
                print("  Not enough paired data to perform paired statistical tests.")
                continue
            if len(data_method1) != len(data_method2):
                print("  Paired data length mismatch.")
                continue
            
            # Calculate the differences between paired observations
            differences = data_method1 - data_method2
            
            # Normality test on the differences using Shapiro-Wilk test
            stat_diff, p_diff = shapiro(differences)
            
            normal_diff = p_diff > alpha
            
            if normal_diff:
                # Use paired t-test
                stat_paired, p_value_paired = ttest_rel(data_method1, data_method2)
                test_name_paired = "Paired t-test"
            else:
                # Use Wilcoxon signed-rank test
                stat_paired, p_value_paired = wilcoxon(data_method1, data_method2)
                test_name_paired = "Wilcoxon signed-rank test"
            
            # Interpret the statistical test result
            if p_value_paired > alpha:
                result_paired = "No significant difference"
            else:
                result_paired = "Significant difference"
            
            # Append the paired test results to the list
            stat_results.append({
                'Metric': metric,
                'Muscle': muscle,
                'Methods Compared': f"{method1} vs {method2}",
                'Test': test_name_paired,
                'Statistic': stat_paired,
                'p-value': p_value_paired,
                'Result': result_paired
            })
            
            # Print paired test results
            print(f"  {test_name_paired}: Statistic={stat_paired:.4f}, p-value={p_value_paired:.4f}")
            print(f"  Result: {result_paired} between {method1} and {method2} for muscle '{muscle}' (paired test).")
            
# Convert the results list to a DataFrame
df_results = pd.DataFrame(stat_results)

# Display the statistical results DataFrame
print("\nStatistical Test Summary:")
print(df_results)

# Create boxplots to visualize the differences
# Use the specified colors
palette = {'Original': '#3274a1', 'Binary': '#e1812c', 'Muscle Specific': '#3a923a'}

for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='class_gt',
        y=metric,
        hue='Method',
        data=df_combined,
        palette=palette
    )
    plt.title(f'Comparison of {metric.capitalize()} Between Methods')
    plt.xlabel('Class GT')
    plt.ylabel(metric.capitalize())
    plt.legend(title='Method')
    plt.xticks()  # Tilt class names by 45 degrees
    plt.tight_layout()
    # Save the plot at 300 dpi
    plt.savefig(f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/BOXPLOT/{metric}_comparison.png', dpi=300)
    # plt.show()

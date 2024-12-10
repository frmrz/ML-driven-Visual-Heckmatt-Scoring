import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def get_file_paths(base_path, iteration_step=5000, max_iteration=65000):
    """
    Retrieves the file paths for predicted and ground truth segmentation maps.
    """
    file_paths = []
    for m in range(0, max_iteration + 1, iteration_step):
        pred_dir = f"{base_path}/checkpoints/dp_gan_thyroid_B_512_v0/segmentation/iter_{m}/pred"
        gt_dir = f"{base_path}/datasets/B_768/validation/seg_maps_512"
        for file in os.listdir(pred_dir):
            file_paths.append((os.path.join(pred_dir, file), os.path.join(gt_dir, file), m))
    return file_paths

def read_image(path):
    """
    Reads an image from a given path and converts it to a numpy array.
    """
    with Image.open(path) as img:
        return np.array(img)

def calculate_iou(pred, gt, num_classes):
    """
    Calculates the IoU for each class, assigning nan for missing labels.
    """
    iou_data = []
    for cls in range(num_classes):
        if cls not in gt:
            iou_data.append(np.nan)
            continue
        intersection = np.logical_and(pred == cls, gt == cls)
        union = np.logical_or(pred == cls, gt == cls)
        union_sum = np.sum(union)
        iou = np.sum(intersection) / union_sum if union_sum != 0 else np.nan
        iou_data.append(iou)
    return iou_data

def main():
    base_path = '/media/francesco/DEV001/PROJECT-KERNEL/DP_GAN'
    num_classes = 9  # Adjust based on actual number of classes
    label_names = ['Internal Background', 'Connective Tissue', 'Trachea', 
                   'Thyroid Gland', 'Blood Vessels', 'Dermal', 
                   'Muscle Tissue', 'Thyroid Nodule', 'External Background']
    file_paths = get_file_paths(base_path)
    iou_data = []

    for pred_path, gt_path, iteration in file_paths:
        print(f"Processing {pred_path}...")
        pred = read_image(pred_path)
        gt = read_image(gt_path)
        ious = calculate_iou(pred, gt, num_classes)
        for cls, iou in enumerate(ious):
            iou_data.append([os.path.basename(pred_path), cls, iteration, iou])

    df = pd.DataFrame(iou_data, columns=['Image', 'Label', 'Checkpoint', 'IoU'])
    save_path = '/media/francesco/DEV001/PROJECT-KERNEL/DP_GAN/checkpoints/dp_gan_thyroid_B_512_v0/metrics_and_features/'
    df.to_csv(f'{save_path}iou_results.csv', index=False)

    plt.figure()
    for label in range(num_classes):
        label_df = df[df['Label'] == label].dropna()
        avg_iou = label_df.groupby('Checkpoint')['IoU'].mean()
        std_iou = label_df.groupby('Checkpoint')['IoU'].std()

        plt.plot(avg_iou.index, avg_iou, label=label_names[label])
        # plt.fill_between(std_iou.index, avg_iou - std_iou, avg_iou + std_iou, alpha=0.3)

    # Calculate and plot macro average IoU and its standard deviation
    macro_avg_iou = df.groupby('Checkpoint')['IoU'].mean()
    macro_std_iou = df.groupby('Checkpoint')['IoU'].std()
    plt.plot(macro_avg_iou.index, macro_avg_iou, label='Macro Average', color='black', linewidth=2)
    # plt.fill_between(macro_std_iou.index, macro_avg_iou - macro_std_iou, macro_avg_iou + macro_std_iou, color='grey', alpha=0.2)

    plt.xlabel('Checkpoint')
    plt.ylabel('IoU')
    plt.title('IoU  Macro Average')
    plt.legend()
    plt.savefig(f'{save_path}all_labels_iou_plot.png')
    plt.close()

    print("IoU calculation and plotting completed.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

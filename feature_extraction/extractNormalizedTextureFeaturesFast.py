import numpy as np
from PIL import Image
from skimage import morphology
import os
import matplotlib.colors as mcolors

from tqdm import tqdm

from radiomics import featureextractor
import logging

import pandas as pd
import cv2
from scipy.ndimage import label
from scipy.spatial import ConvexHull
import SimpleITK as sitk  # Import SimpleITK for in-memory image handling

import multiprocessing
from functools import partial


def retain_largest_object(mask):
    labeled, num = label(mask)
    sizes = np.bincount(labeled.ravel())

    if len(sizes) > 1:  # Ensure there are labels other than the background
        max_label = np.argmax(sizes[1:]) + 1  # Ignore background label
        mask = (labeled == max_label)

    return mask


def postProcessNetworkOutput(pred, class_labels, class_gt, label_gt):

    unique_labels, counts = np.unique(pred, return_counts=True)
    labels = unique_labels[unique_labels > 0]
    labels_other = labels[labels != label_gt]

    if len(labels_other) > 0:
        pred_binary = pred > 0
        labeled_array, num_features = label(pred_binary)

        if num_features == 1:
            pred[pred > 0] = label_gt
        else:
            sizes = np.bincount(labeled_array.ravel())
            sizes[0] = 0  # Background size
            max_label = np.argmax(sizes)

            biggest_object = (labeled_array == max_label)
            other_objects = pred_binary & (~biggest_object)

            selem = morphology.disk(7)
            dilated_biggest_object = morphology.dilation(biggest_object, selem)

            added_pixels = dilated_biggest_object & biggest_object
            overlap = dilated_biggest_object & other_objects
            overlap_added = added_pixels & other_objects

            overlap_area = np.sum(overlap)
            overlap_area_added = np.sum(overlap_added)

            overlap_percentage = overlap_area / (np.sum(added_pixels) + np.finfo(float).eps)
            overlap_percentage_added = overlap_area_added / (np.sum(added_pixels) + np.finfo(float).eps)

            if overlap_percentage > 0.1:
                flag_overlap = 1
            else:
                flag_overlap = 0

            if overlap_percentage_added > 0.1:
                flag_overlap_added = 1
            else:
                flag_overlap_added = 0

            if flag_overlap == 1:
                pred[pred > 0] = label_gt
            else:
                pred[pred != label_gt] = 0

    unique_labels, counts = np.unique(pred, return_counts=True)
    labels = unique_labels[unique_labels > 0]
    counts = counts[unique_labels > 0]

    if len(labels) > 1:
        oh_pred = (np.arange(len(class_labels)+1) == pred[..., None]).astype(int) > 0
        oh_pred_original = oh_pred.copy()

        for i in labels:
            selem = morphology.disk(2)
            oh_pred[..., i] = morphology.dilation(oh_pred[..., i], selem)
            oh_pred[..., i] = morphology.remove_small_holes(oh_pred[..., i], 100000)
            msk_cv = oh_pred[..., i].astype(np.uint8)
            contours, _ = cv2.findContours(msk_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and len(contours) > 0:
                all_points = np.vstack(contours).squeeze()
                if len(all_points.shape) == 2 and all_points.shape[0] >= 3:
                    hull = ConvexHull(all_points)
                    hull_points = all_points[hull.vertices]
                    blank_mask = np.zeros(oh_pred[..., i].shape, dtype=np.uint8)
                    oh_pred[..., i] = cv2.fillPoly(blank_mask, pts=[hull_points], color=(255)) > 0

        for i in labels:
            mask1 = oh_pred[..., i]
            sum_mask1 = oh_pred_original[..., i].sum()

            for j in labels:
                if j > i:
                    mask2 = oh_pred[..., j]
                    sum_mask2 = oh_pred_original[..., j].sum()
                    intersection = np.logical_and(mask1, mask2).sum()
                    union = np.logical_or(mask1, mask2).sum()
                    iou = intersection / union if union != 0 else 0

                    if iou > 0.6:
                        if sum_mask1 > sum_mask2:
                            pred[pred == j] = i
                        else:
                            pred[pred == i] = j

        oh_pred = (np.arange(len(class_labels)+1) == pred[..., None]).astype(int) > 0

        for i in labels:
            selem = morphology.disk(2)
            oh_pred[..., i] = morphology.dilation(oh_pred[..., i], selem)
            oh_pred[..., i] = morphology.remove_small_holes(oh_pred[..., i], 100000)
            oh_pred[..., i] = retain_largest_object(oh_pred[..., i])
            oh_pred[..., i] = morphology.erosion(oh_pred[..., i], selem)

        pred = np.argmax(oh_pred, axis=-1)
        unique_labels, counts = np.unique(pred, return_counts=True)
        dominant_label = labels[np.argmax(counts)]
        class_pred = classes[dominant_label]
        pred_out = pred

        return pred_out, class_pred

    elif len(labels) == 1:
        dominant_label = labels[np.argmax(counts)]
        seg_map = (pred > 0).astype(np.uint8)
        selem = morphology.disk(3)
        seg_map = morphology.dilation(morphology.erosion(seg_map, selem), selem)
        seg_map = morphology.opening(seg_map, selem)
        seg_map = morphology.closing(seg_map, selem)
        seg_map = retain_largest_object(seg_map)
        pred_out = seg_map * dominant_label
        class_pred = classes[dominant_label]

        return pred_out, class_pred

    else:
        dominant_label = 0
        class_pred = classes[dominant_label]
        pred_out = pred

        return pred_out, class_pred


def process_file(file, fold, pred_fold, gt_fold, img_fold, muscle, classes, class_labels):
    import numpy as np
    import cv2
    from scipy.ndimage import label
    import SimpleITK as sitk
    from radiomics import featureextractor
    import os
    temp = dict()

    temp['Fold'] = fold
    temp['Muscle'] = muscle  # Add current muscle to the summary

    # Load image, ground truth and prediction
    img_PIL = Image.open(os.path.join(img_fold, file))
    gt_PIL = Image.open(os.path.join(gt_fold, file))
    pred_PIL = Image.open(os.path.join(pred_fold, file))

    img = np.array(img_PIL)
    
    # if img has 3 channels, keep only the first channel
    if len(img.shape) == 3:
        img = img[..., 0]
                
    gt = np.array(gt_PIL)
    pred = np.array(pred_PIL)

    # Ensure all arrays have the same shape
    if img.shape != gt.shape or img.shape != pred.shape:
        min_height = min(img.shape[0], gt.shape[0], pred.shape[0])
        min_width = min(img.shape[1], gt.shape[1], pred.shape[1])
        img = img[:min_height, :min_width]
        gt = gt[:min_height, :min_width]
        pred = pred[:min_height, :min_width]

    # Process pred to have a unique prediction
    unique_labels_gt, counts_gt = np.unique(gt, return_counts=True)

    # Exclude label 0 (background)
    if len(unique_labels_gt) > 1:
        label_gt = unique_labels_gt[1]
        class_gt = classes[label_gt]
    else:
        label_gt = 0
        class_gt = classes[label_gt]

    pred_out, class_pred = postProcessNetworkOutput(pred, class_labels, class_gt, label_gt)

    # Find index of class_pred in class_labels
    label_pred = np.where(np.array(classes) == class_pred)[0][0]

    # Check number of labels in pred_out
    unique_labels_pred, counts_pred = np.unique(pred_out, return_counts=True)

    if label_gt in unique_labels_pred:
        class_pred = class_gt
        label_pred = label_gt

    # Initialize SimpleITK images
    if len(img.shape) == 3:
        img = img[:, :, 0]
    sitk_image = sitk.GetImageFromArray(img)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

    # Set consistent spacing, origin, and direction
    spacing = [1.0, 1.0]
    origin = [0.0, 0.0]
    direction = [1.0, 0.0, 0.0, 1.0]  # 2D identity matrix flattened

    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)

    # Create feature extractor inside the function to avoid pickling issues
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName('Original')
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.settings['additionalInfo'] = False

    # For each label in the prediction
    results = []
    for i in unique_labels_pred:
        temp_copy = temp.copy()
        if len(unique_labels_pred) > 1:
            if i == label_pred and i != 0:
                # Prepare ground truth mask
                gt_bw = gt > 0
                gt_mask = (gt_bw * 255).astype(np.uint8)
                sitk_gt_mask = sitk.GetImageFromArray(gt_mask)
                sitk_gt_mask.SetSpacing(spacing)
                sitk_gt_mask.SetOrigin(origin)
                sitk_gt_mask.SetDirection(direction)

                # Prepare predicted mask
                pred_out_copy = pred_out.copy()
                pred_out_copy[pred_out_copy != i] = 0
                pred_bw = pred_out_copy > 0
                pred_mask = (pred_bw * 255).astype(np.uint8)
                sitk_pred_mask = sitk.GetImageFromArray(pred_mask)
                sitk_pred_mask.SetSpacing(spacing)
                sitk_pred_mask.SetOrigin(origin)
                sitk_pred_mask.SetDirection(direction)

                # Compute metrics
                TP = np.sum(np.logical_and(gt_bw, pred_bw))
                TN = np.sum(np.logical_not(np.logical_or(gt_bw, pred_bw)))
                FP = np.sum(np.logical_and(np.logical_not(gt_bw), pred_bw))
                FN = np.sum(np.logical_and(gt_bw, np.logical_not(pred_bw)))
                iou_score = TP / (TP + FP + FN + np.finfo(float).eps)
                precision = TP / (TP + FP + np.finfo(float).eps)
                recall = TP / (TP + FN + np.finfo(float).eps)

                # Store values
                temp_copy['File'] = file
                temp_copy['subject'] = file.split('_')[0]
                temp_copy['muscle_code'] = file.split('_')[1]
                temp_copy['side'] = file.split('_')[2]

                temp_copy['class_gt'] = class_gt
                temp_copy['class_pred'] = class_pred
                temp_copy['iou'] = iou_score
                temp_copy['prec'] = precision
                temp_copy['rec'] = recall

                try:
                    features = extractor.execute(sitk_image, sitk_gt_mask, label=255)
                    temp_copy['features_img_gt'] = {str(key): str(value) for key, value in features.items()}
                except Exception as e:
                    temp_copy['features_img_gt'] = 'mask not found'

                try:
                    # Prepare negative ground truth mask
                    kernel_size = 20
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    dilated_mask = cv2.dilate(gt.astype(np.uint8), kernel, iterations=1)
                    neg_seg_map = np.logical_not(dilated_mask).astype(np.uint8) * 255

                    border_width = int(0.15 * min(gt.shape[:2]))
                    offset = int(0.10 * min(gt.shape[:2]))

                    neg_seg_map[:border_width, :] = 0
                    neg_seg_map[-border_width:, :] = 0
                    neg_seg_map[:, :border_width] = 0
                    neg_seg_map[:, -border_width:] = 0

                    labeled_array, num_features = label(gt_bw)
                    if labeled_array.size == 0 or num_features == 0:
                        object_centroid = [0, 0]
                    else:
                        coords = np.column_stack(np.where(labeled_array > 0))
                        if coords.size == 0:
                            object_centroid = [0, 0]
                        else:
                            object_centroid = coords.mean(axis=0)

                    neg_seg_map[:int(object_centroid[0] + offset), :] = 0
                    sitk_neg_gt_mask = sitk.GetImageFromArray(neg_seg_map)
                    sitk_neg_gt_mask.SetSpacing(spacing)
                    sitk_neg_gt_mask.SetOrigin(origin)
                    sitk_neg_gt_mask.SetDirection(direction)

                    features = extractor.execute(sitk_image, sitk_neg_gt_mask, label=255)
                    temp_copy['features_img_gt_not'] = {str(key): str(value) for key, value in features.items()}
                except Exception as e:
                    temp_copy['features_img_gt_not'] = 'mask not found'

                try:
                    features = extractor.execute(sitk_image, sitk_pred_mask, label=255)
                    temp_copy['features_img_pred'] = {str(key): str(value) for key, value in features.items()}
                except Exception as e:
                    temp_copy['features_img_pred'] = 'mask not found'

                try:
                    # Prepare negative predicted mask
                    dilated_mask_pred = cv2.dilate(pred.astype(np.uint8), kernel, iterations=1)
                    neg_seg_map_pred = np.logical_not(dilated_mask_pred).astype(np.uint8) * 255

                    neg_seg_map_pred[:border_width, :] = 0
                    neg_seg_map_pred[-border_width:, :] = 0
                    neg_seg_map_pred[:, :border_width] = 0
                    neg_seg_map_pred[:, -border_width:] = 0

                    labeled_array_pred, num_features_pred = label(pred_bw)
                    if labeled_array_pred.size == 0 or num_features_pred == 0:
                        object_centroid_pred = [0, 0]
                    else:
                        coords_pred = np.column_stack(np.where(labeled_array_pred > 0))
                        if coords_pred.size == 0:
                            object_centroid_pred = [0, 0]
                        else:
                            object_centroid_pred = coords_pred.mean(axis=0)

                    neg_seg_map_pred[:int(object_centroid_pred[0] + offset), :] = 0
                    sitk_neg_pred_mask = sitk.GetImageFromArray(neg_seg_map_pred)
                    sitk_neg_pred_mask.SetSpacing(spacing)
                    sitk_neg_pred_mask.SetOrigin(origin)
                    sitk_neg_pred_mask.SetDirection(direction)

                    features = extractor.execute(sitk_image, sitk_neg_pred_mask, label=255)
                    temp_copy['features_img_pred_not'] = {str(key): str(value) for key, value in features.items()}
                except Exception as e:
                    temp_copy['features_img_pred_not'] = 'mask not found'

                results.append(temp_copy)

            elif i != 0:
                # Modify file name
                file_new = file.replace(class_pred, classes[i])
                slice_number = int(file_new.split('_')[3].split('.')[0])
                file_new = file_new.replace(file_new.split('_')[3], str(slice_number + 90))

                # Prepare predicted mask
                pred_out_copy = pred_out.copy()
                pred_out_copy[pred_out_copy != i] = 0
                pred_bw = pred_out_copy > 0
                pred_mask = (pred_bw * 255).astype(np.uint8)
                sitk_pred_mask = sitk.GetImageFromArray(pred_mask)
                sitk_pred_mask.SetSpacing(spacing)
                sitk_pred_mask.SetOrigin(origin)
                sitk_pred_mask.SetDirection(direction)

                # Store values
                temp_copy['File'] = file_new
                temp_copy['subject'] = file_new.split('_')[0]
                temp_copy['muscle_code'] = file_new.split('_')[1]
                temp_copy['side'] = file_new.split('_')[2]

                temp_copy['class_gt'] = class_gt
                temp_copy['class_pred'] = class_pred
                temp_copy['iou'] = np.nan
                temp_copy['prec'] = np.nan
                temp_copy['rec'] = np.nan

                temp_copy['features_img_gt'] = 'mask not found'
                temp_copy['features_img_gt_not'] = 'mask not found'

                try:
                    features = extractor.execute(sitk_image, sitk_pred_mask, label=255)
                    temp_copy['features_img_pred'] = {str(key): str(value) for key, value in features.items()}
                except Exception as e:
                    temp_copy['features_img_pred'] = 'mask not found'

                try:
                    # Prepare negative predicted mask
                    dilated_mask_pred = cv2.dilate(pred.astype(np.uint8), kernel, iterations=1)
                    neg_seg_map_pred = np.logical_not(dilated_mask_pred).astype(np.uint8) * 255

                    neg_seg_map_pred[:border_width, :] = 0
                    neg_seg_map_pred[-border_width:, :] = 0
                    neg_seg_map_pred[:, :border_width] = 0
                    neg_seg_map_pred[:, -border_width:] = 0

                    labeled_array_pred, num_features_pred = label(pred_bw)
                    if labeled_array_pred.size == 0 or num_features_pred == 0:
                        object_centroid_pred = [0, 0]
                    else:
                        coords_pred = np.column_stack(np.where(labeled_array_pred > 0))
                        if coords_pred.size == 0:
                            object_centroid_pred = [0, 0]
                        else:
                            object_centroid_pred = coords_pred.mean(axis=0)

                    neg_seg_map_pred[:int(object_centroid_pred[0] + offset), :] = 0
                    sitk_neg_pred_mask = sitk.GetImageFromArray(neg_seg_map_pred)
                    sitk_neg_pred_mask.SetSpacing(spacing)
                    sitk_neg_pred_mask.SetOrigin(origin)
                    sitk_neg_pred_mask.SetDirection(direction)

                    features = extractor.execute(sitk_image, sitk_neg_pred_mask, label=255)
                    temp_copy['features_img_pred_not'] = {str(key): str(value) for key, value in features.items()}
                except Exception as e:
                    temp_copy['features_img_pred_not'] = 'mask not found'

                results.append(temp_copy)
        else:
            # Prepare ground truth mask
            gt_bw = gt > 0
            gt_mask = (gt_bw * 255).astype(np.uint8)
            sitk_gt_mask = sitk.GetImageFromArray(gt_mask)
            sitk_gt_mask.SetSpacing(spacing)
            sitk_gt_mask.SetOrigin(origin)
            sitk_gt_mask.SetDirection(direction)

            # Prepare negative ground truth mask
            kernel_size = 20
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_mask = cv2.dilate(gt.astype(np.uint8), kernel, iterations=1)
            neg_seg_map = np.logical_not(dilated_mask).astype(np.uint8) * 255

            border_width = int(0.15 * min(gt.shape[:2]))
            offset = int(0.10 * min(gt.shape[:2]))

            neg_seg_map[:border_width, :] = 0
            neg_seg_map[-border_width:, :] = 0
            neg_seg_map[:, :border_width] = 0
            neg_seg_map[:, -border_width:] = 0

            labeled_array, num_features = label(gt_bw)
            if labeled_array.size == 0 or num_features == 0:
                object_centroid = [0, 0]
            else:
                coords = np.column_stack(np.where(labeled_array > 0))
                if coords.size == 0:
                    object_centroid = [0, 0]
                else:
                    object_centroid = coords.mean(axis=0)

            neg_seg_map[:int(object_centroid[0] + offset), :] = 0
            sitk_neg_gt_mask = sitk.GetImageFromArray(neg_seg_map)
            sitk_neg_gt_mask.SetSpacing(spacing)
            sitk_neg_gt_mask.SetOrigin(origin)
            sitk_neg_gt_mask.SetDirection(direction)

            # Store values
            temp_copy['File'] = file
            temp_copy['subject'] = file.split('_')[0]
            temp_copy['muscle_code'] = file.split('_')[1]
            temp_copy['side'] = file.split('_')[2]

            temp_copy['class_gt'] = class_gt
            temp_copy['class_pred'] = 'background'
            temp_copy['iou'] = 0
            temp_copy['prec'] = 0
            temp_copy['rec'] = 0

            try:
                features = extractor.execute(sitk_image, sitk_gt_mask, label=255)
                temp_copy['features_img_gt'] = {str(key): str(value) for key, value in features.items()}
            except Exception as e:
                temp_copy['features_img_gt'] = 'mask not found'

            try:
                features = extractor.execute(sitk_image, sitk_neg_gt_mask, label=255)
                temp_copy['features_img_gt_not'] = {str(key): str(value) for key, value in features.items()}
            except Exception as e:
                temp_copy['features_img_gt_not'] = 'mask not found'

            temp_copy['features_img_pred'] = 'mask not found'
            temp_copy['features_img_pred_not'] = 'mask not found'

            results.append(temp_copy)

    return results


# Prepare parameters for feature extraction
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)

# Define base preds_dirs with a placeholder for muscle name
base_preds_dirs_template = [
    "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_{muscle}/pred",
    "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_{muscle}/pred",
    "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_{muscle}/pred",
    "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_{muscle}/pred",
    "/home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_{muscle}/pred"
]

gt_dirs = [
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f0/labels/testing",
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f1/labels/testing",
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f2/labels/testing",
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f3/labels/testing",
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f4/labels/testing"
]

image_dirs = [
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f0/images/testing",
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f1/images/testing",
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f2/images/testing",
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f3/images/testing",
    "/home/francesco/Desktop/POLI/RADBOUD/DATA/DEVELOPMENT/FSHD_v3_f4/images/testing"
]

net = 'knet_swin_mod'
experiment = 'muscle_specific'

# Define class and palette for better visualization
classes = [
    'background',
    'Biceps_brachii',
    'Deltoideus',
    'Depressor_anguli_oris',
    'Digastricus',
    'Gastrocnemius_medial_head',
    'Geniohyoideus',
    'Masseter',
    'Mentalis',
    'Orbicularis_oris',
    'Rectus_abdominis',
    'Rectus_femoris',
    'Temporalis',
    'Tibialis_anterior',
    'Trapezius',
    'Vastus_lateralis',
    'Zygomaticus'
]

# Exclude 'background' from the list
muscle_names = classes[1:]

# Convert the list into a single string, separated by comma and space
muscle_names_str = ', '.join(muscle_names)

print("Muscles to be processed:", muscle_names_str)

class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

color_names = [
    'black', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white',
    'purple', 'lime', 'teal', 'navy', 'grey', 'maroon', 'olive', 'brown',
    'coral'
]

cmap_muscle = mcolors.ListedColormap(color_names)

# Initialize summary list to collect results from all muscles
summary = []

# Load missing filenames from txt file missing_filenames.txt
# missing_filenames = np.loadtxt('/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/missing_filenames.txt', dtype=str)

# Loop over each muscle
for muscle in muscle_names:

    print(f"\nProcessing muscle: {muscle}\n")

    # Update preds_dirs for the current muscle by formatting the template paths
    preds_dirs = [path.format(muscle=muscle) for path in base_preds_dirs_template]

    fold = 0  # Reset fold counter for each muscle

    # Loop over each fold
    for pred_fold, gt_fold, img_fold in zip(preds_dirs, gt_dirs, image_dirs):

        print(f"Processing fold {fold} for muscle {muscle}\n")

        filenames = os.listdir(pred_fold)

        # Prepare the partial function with fixed arguments
        partial_process_file = partial(
            process_file,
            fold=fold,
            pred_fold=pred_fold,
            gt_fold=gt_fold,
            img_fold=img_fold,
            muscle=muscle,
            classes=classes,
            class_labels=class_labels
        )

        # Use multiprocessing Pool
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(partial_process_file, filenames), total=len(filenames), desc=f"Processing files in fold {fold}"))

        # Flatten the list of lists
        for res in results:
            summary.extend(res)

        fold += 1

# Convert the summary list of dictionaries to a DataFrame
df = pd.DataFrame().from_dict(summary)

# Save the DataFrame to a single Excel file
output_excel_path = f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/segmentation_summary_{net}_{experiment}.xlsx'
df.to_excel(output_excel_path, index=False)
print(f"\nSummary Excel file saved to: {output_excel_path}")

# Optionally, save the DataFrame to a JSON file as well
output_json_path = f'/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/segmentation_summary_{net}_{experiment}.json'
df.to_json(output_json_path, indent=4)
print(f"Summary JSON file saved to: {output_json_path}")

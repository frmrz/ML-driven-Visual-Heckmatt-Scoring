import numpy as np
from PIL import Image, ImageDraw
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
import matplotlib.pyplot as plt
from skimage import measure

def retain_largest_object(mask):
    labeled, num = label(mask)
    sizes = np.bincount(labeled.ravel())

    if len(sizes) > 1:  # Ensure there are labels other than the background
        max_label = np.argmax(sizes[1:]) + 1  # Ignore background label
        mask = (labeled == max_label)

    # If sizes is empty or only contains the background, return the input mask as it is

    return mask

### DEFINE VISUALIZATION FUNCTIONS

def overlay_contours(img_rgb, gt, neg_seg_map, save_path):
    
    # Find contours in the binary segmentation maps
    gt_contours = measure.find_contours(gt, 0.5, fully_connected='high')
    neg_seg_contours = measure.find_contours(neg_seg_map, 0.5, fully_connected='high')

    img_rgb = Image.fromarray(img_rgb)
    # Create a new image with the same dimensions as 'img_rgb'
    overlay_img = Image.new('RGBA', img_rgb.size)
    overlay_draw = ImageDraw.Draw(overlay_img)

    # Draw green contours from 'gt'
    for contour in gt_contours:
        contour = np.fliplr(contour)  # Reverses the x and y coordinates
        overlay_draw.line(contour.ravel().tolist(), fill=(0, 255, 0), width=5)

    # Draw red contours from 'neg_seg_map'
    for contour in neg_seg_contours:
        contour = np.fliplr(contour)  # Reverses the x and y coordinates
        overlay_draw.line(contour.ravel().tolist(), fill=(255, 0, 0), width=5)

    # Create a transparent image with the same dimensions as 'img_rgb'
    transparent_img = Image.new('RGBA', img_rgb.size, (0, 0, 0, 0))

    # Composite the original image and the overlay image
    final_img = Image.alpha_composite(transparent_img, img_rgb.convert('RGBA'))
    final_img = Image.alpha_composite(final_img, overlay_img)

    # Convert back to RGB mode
    final_img = final_img.convert('RGB')

    # Save the final image
    final_img.save(save_path)
    
def postProcessNetworkOutput(pred, class_labels, class_gt, label_gt):
    
    unique_labels, counts = np.unique(pred, return_counts=True)  # Exclude background
    
    labels = unique_labels[unique_labels > 0]
    counts = counts[unique_labels > 0]
    
    # ===========================
    # Rev1 - modified post-processing
    # ===========================

    # Get unique labels in the prediction
    unique_labels, counts = np.unique(pred, return_counts=True)

    # Exclude background (label 0)
    labels = unique_labels[unique_labels > 0]

    # Exclude label_gt from labels
    labels_other = labels[labels != label_gt]

    # 1) Look if any other label other than 0 and label_gt is present in the prediction
    if len(labels_other) > 0:
        # Create a binary mask for all labels greater than 0
        pred_binary = pred > 0

        # Label connected components in the binary prediction
        labeled_array, num_features = label(pred_binary)

        # 2) Check how many connected objects are present in the prediction
        if num_features == 1:
            # 2.a) If only one object is present, correct the label assigning the pixels to the label_gt class
            pred[pred > 0] = label_gt
        else:
            # 2.b) If more than one object is present, check if those are two distinct objects

            # Find the biggest object (largest connected component)
            sizes = np.bincount(labeled_array.ravel())
            sizes[0] = 0  # Background size
            max_label = np.argmax(sizes)

            # Create masks for the biggest object and other objects
            biggest_object = (labeled_array == max_label)
            other_objects = pred_binary & (~biggest_object)

            # Dilate the biggest object
            selem = morphology.disk(7)  # Adjust the size as needed
            dilated_biggest_object = morphology.dilation(biggest_object, selem)
            
            # Create mask of the added pixels in the biggest object
            added_pixels = dilated_biggest_object & biggest_object

            # Compute overlap between dilated biggest object and other objects
            overlap = dilated_biggest_object & other_objects
            
            # Compute overlap between the added pixels and other objects
            overlap_added = added_pixels & other_objects

            # Calculate overlap area
            overlap_area = np.sum(overlap)
            
            # Calculate overlap area of added pixels
            overlap_area_added = np.sum(overlap_added)

            # Calculate area of smaller object(s)
            smaller_objects_area = np.sum(other_objects)

            # Calculate the overlap percentage
            overlap_percentage = overlap_area / (np.sum(added_pixels) + np.finfo(float).eps)
            
            # calculate the overlap percentage of added pixels
            overlap_percentage_added = overlap_area_added / (np.sum(added_pixels) + np.finfo(float).eps)

            print(f'Overlap percentage: {overlap_percentage:.2f}')
            print(f'Overlap percentage added: {overlap_percentage_added:.2f}')
            
            # Check if the overlap exceeds 10%
            if overlap_percentage > 0.1:
                flag_overlap = 1
            else:
                flag_overlap = 0
                
            # Check if the overlap of added pixels exceeds 10%
            if overlap_percentage_added > 0.1:
                flag_overlap_added = 1
            else:
                flag_overlap_added = 0

            # 2.c) Apply corrections based on flag_overlap
            if flag_overlap == 1:
                # Assign all pixels of the prediction to the label_gt class
                pred[pred > 0] = label_gt
            else:
                # Set to 0 the pixels of pred that are not label_gt
                pred[pred != label_gt] = 0

    # ===========================
    # Original Post-processing
    # ===========================
    unique_labels, counts = np.unique(pred, return_counts=True)  # Exclude background
    
    labels = unique_labels[unique_labels > 0]
    counts = counts[unique_labels > 0]
    
    if len(labels) > 1:
        
        # one-hot encode the segmentation map
        oh_pred = (np.arange(len(class_labels)+1) == pred[...,None]).astype(int)>0
        oh_pred_original = (np.arange(len(class_labels)+1) == pred[...,None]).astype(int)>0

        # STEP 1: look for overlapped muscles and merge them on the dominant one updating pred
        for i in labels:
            # Create a disk structuring element
            selem = morphology.disk(2)

            # Apply dilation to avoid line breaks
            oh_pred[..., i] = morphology.dilation(oh_pred[..., i], selem)

            # Fill big holes in mask
            oh_pred[..., i] = morphology.remove_small_holes(oh_pred[..., i], 100000)

            # Find contour
            msk_cv = oh_pred[..., i].astype(np.uint8)
            contours, _ = cv2.findContours(msk_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Combine all contour points
            all_points = np.vstack(contours).squeeze()

            # Compute the convex hull
            hull = ConvexHull(all_points)
            hull_points = all_points[hull.vertices]
            blank_mask = np.zeros(oh_pred[..., i].shape, dtype=np.uint8)
            oh_pred[..., i] = cv2.fillPoly(blank_mask, pts=[hull_points], color=(255))>0

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

                    # If iou is bigger than 0.6, in the pred mask set the smaller label to the bigger one
                    if iou > 0.6:
                        if sum_mask1 > sum_mask2:
                            pred[pred == j] = i
                        else:
                            pred[pred == i] = j

        # STEP 2: for each label, look for the biggest connected component and remove all the others, then update pred
        oh_pred = (np.arange(len(class_labels)+1) == pred[...,None]).astype(int)>0

        for i in labels:
            # Create a disk structuring element
            selem = morphology.disk(2)

            # Apply dilation to avoid line breaks
            oh_pred[..., i] = morphology.dilation(oh_pred[..., i], selem)

            # Fill big holes in mask
            oh_pred[..., i] = morphology.remove_small_holes(oh_pred[..., i], 100000)

            # Retain only the biggest connected component
            oh_pred[..., i] = retain_largest_object(oh_pred[..., i])

            # Apply erosion to restore original shape
            oh_pred[..., i] = morphology.erosion(oh_pred[..., i], selem)

        # reconstruct pred from one-hot maps
        pred = np.argmax(oh_pred, axis=-1)

        # STEP 3: assign class prediction to the dominant label in pred
        unique_labels, counts = np.unique(pred, return_counts=True)  # Exclude background

        dominant_label = labels[np.argmax(counts)]
        class_pred = classes[dominant_label]
        pred_out = pred

        return pred_out, class_pred
            
    elif len(labels) == 1:
            
            # Find the dominant label
            dominant_label = labels[np.argmax(counts)]

            # Binary segmentation: set the dominant label to 1, all other pixels to 0
            seg_map = (pred > 0).astype(np.uint8)
            
            # Apply morphological operations
            # Create a disk structuring element
            selem = morphology.disk(3)
            
            # Apply erosion and dilation to smooth boundaries
            seg_map = morphology.dilation(morphology.erosion(seg_map, selem), selem)
            
            # Apply opening to remove small objects
            seg_map = morphology.opening(seg_map, selem)
            
            # Apply closing to fill small holes
            seg_map = morphology.closing(seg_map, selem)

            # Retain only the biggest connected component
            seg_map = retain_largest_object(seg_map)
            
            pred_out = seg_map * dominant_label
            class_pred = classes[dominant_label]

            return pred_out, class_pred

    else:
        dominant_label = 0

        class_pred = classes[dominant_label]
        pred_out = pred

        return pred_out, class_pred
    
    # Additional plotting code is commented out for clarity

### PREPARE PARAMETERS FOR FEATURE EXTRACTION

# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)

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
# define class and palette for better visualization
classes = [
    'background',
    'Biceps_brachii', # 001 - 1 for label
    'Deltoideus', # 002
    'Depressor_anguli_oris', # 003
    'Digastricus', # 004
    'Gastrocnemius_medial_head', # 008
    'Geniohyoideus', # 009
    'Masseter', # 011
    'Mentalis', # 012
    'Orbicularis_oris', # 013
    'Rectus_abdominis', # 015
    'Rectus_femoris', # 016
    'Temporalis', # 017
    'Tibialis_anterior', # 018
    'Trapezius', # 019
    'Vastus_lateralis', # 020
    'Zygomaticus'  # 021
]

# Exclude 'background' from the list
muscle_names = classes[1:]

# Convert the list into a single string, separated by comma and space
muscle_names_str = ', '.join(muscle_names)

print("Muscles to be processed:", muscle_names_str)

class_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

color_names = [
    'black', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 
    'purple', 'lime', 'teal', 'navy', 'grey', 'maroon', 'olive', 'brown', 
    'coral'
]

cmap_muscle = mcolors.ListedColormap(color_names)

# Initialize summary list to collect results from all muscles
summary = []

##########
###### POST PROCESS NETWORK OUTPUT AND EXTRACT METRICS AND FEATURES
##########

# Load missing filenames from txt file missing_filenames.txt
missing_filenames = np.loadtxt('/home/francesco/Desktop/POLI/RADBOUD/RESULTS/EXCEL/missing_filenames.txt', dtype=str)

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
        
        for file in tqdm(filenames, desc=f"Processing files in fold {fold}"):
    
            print(f"Processing file {file} \n")    
            temp = dict()
    
            ##### lines for debugging
            # fold = 2
            # file = '02151_015_01_1.png'
            # pred_fold = preds_dirs[fold]
            # gt_fold = gt_dirs[fold]
            # img_fold = image_dirs[fold]
            ##### lines for debugging
    
            temp['Fold'] = fold
            temp['Muscle'] = muscle  # Add current muscle to the summary
    
            # Load image, ground truth and prediction
            img_PIL  = Image.open(os.path.join(img_fold, file))
            
            # Convert the image to a NumPy array
            image_array = np.array(img_PIL)
            img_PIL.save('./temp_img.png')

            gt_PIL   = Image.open(os.path.join(gt_fold, file))
            pred_PIL = Image.open(os.path.join(pred_fold, file))
            zero_PIL = Image.new('L', (512,512))

            img  = np.array(img_PIL)
            gt   = np.array(gt_PIL)
            pred = np.array(pred_PIL)
            
            # Process pred to have a unique prediction
            unique_labels_gt, counts_gt = np.unique(gt, return_counts=True)  # Exclude background
            
            # Exclude label 0 (background)
            if len(unique_labels_gt) > 1:  
                label_gt = unique_labels_gt[1]
            
                class_gt = classes[label_gt]
            else:
                label_gt = 0
                class_gt = classes[label_gt]
            
            pred_out, class_pred = postProcessNetworkOutput(pred, class_labels, class_gt, label_gt)
            
            # find index of class_pred in class_labels
            label_pred = np.where(np.array(classes) == class_pred)[0][0]

            # check number of labels in pred_out
            unique_labels_pred, counts_pred = np.unique(pred_out, return_counts=True)  # Exclude background

            if label_gt in unique_labels_pred:
                class_pred = class_gt
                label_pred = label_gt

            # INIT - From here take into account the possibility to have more than one label in pred_out
            for i in unique_labels_pred:
                if len(unique_labels_pred) > 1:
                        
                    # check if at least one of unique_labels_pred is the same as label_gt
                            
                    # as pred filename use the gt filename if i is equal to the label of class_pred else change the muscle code to the one of i and put 99 as the slice number
                    if i == label_pred:
                        # file = file

                        gt_bw = gt > 0
                        gt_bw_PIL = Image.fromarray(np.uint8(gt_bw*255))
                        gt_bw_PIL.save('./temp_gt_mask.png')

                        # Put everything to zero except the current label on a copy of pred_out
                        pred_out_copy = pred_out.copy()
                        pred_out_copy[pred_out_copy != i] = 0

                        pred_bw = pred_out_copy > 0
                        pred_bw_PIL = Image.fromarray(np.uint8(pred_bw*255))
                        pred_bw_PIL.save('./temp_pred_mask.png')
                    
                        # Dilate the object in the 
                        kernel_size = 20
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

                        dilated_mask = cv2.dilate(gt, kernel, iterations=1)
                        
                        # Obtain the negative segmentation map
                        neg_seg_map = np.logical_not(dilated_mask).astype(np.uint8) * 255
                        
                        # Put 15% of pixels to zero on each border
                        border_width = int(0.15 * min(gt.shape[:2]))
                        offset = int(0.10 * min(gt.shape[:2]))
                        
                        neg_seg_map[:border_width, :] = 0
                        neg_seg_map[-border_width:, :] = 0
                        neg_seg_map[:, :border_width] = 0
                        neg_seg_map[:, -border_width:] = 0

                        # Find the centroid of the original object
                        labeled_array, num_features = label(gt_bw)
                        if labeled_array.size == 0:
                            object_centroid = 0
                        else:
                            coords = np.column_stack(np.where(labeled_array > 0))
                            if coords.size == 0:
                                object_centroid = 0
                            else:
                                object_centroid = coords.mean(axis=0)

                        # Put to zero all pixels above the centroid
                        neg_seg_map[:int(object_centroid[0]+offset), :] = 0

                        # Save the negative segmentation map gt
                        neg_seg_map_PIL = Image.fromarray(neg_seg_map)
                        neg_seg_map_PIL.save('./temp_gt_not_mask.png')   

                        # True Positive (TP): we count the pixels that are 1 in both masks
                        TP = np.sum(np.logical_and(gt_bw, pred_bw))
                        # True Negative (TN): we count the pixels that are 0 in both masks
                        TN = np.sum(np.logical_not(np.logical_or(gt_bw, pred_bw)))
                        # False Positive (FP): we count the pixels that are 1 in the predicted mask but 0 in the ground truth mask
                        FP = np.sum(np.logical_and(np.logical_not(gt_bw), pred_bw))
                        # False Negative (FN): we count the pixels that are 0 in the predicted mask but 1 in the ground truth mask
                        FN = np.sum(np.logical_and(gt_bw, np.logical_not(pred_bw)))
                        # Intersection over Union (IoU)
                        iou_score = TP / (TP + FP + FN + np.finfo(float).eps)
                        # Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
                        precision = TP / (TP + FP + np.finfo(float).eps)
                        # Recall (Sensitivity) - the ratio of correctly predicted positive observations to the all observations in actual class
                        recall = TP / (TP + FN + np.finfo(float).eps)
                        
                        # Store values
                        temp['File'] = file
                        temp['subject'] = file.split('_')[0]
                        temp['muscle_code']  = file.split('_')[1]
                        temp['side']    = file.split('_')[2]

                        temp['class_gt'] = class_gt
                        temp['class_pred'] = class_pred
                        temp['iou'] = iou_score
                        temp['prec'] = precision
                        temp['rec'] = recall
                        
                        try:
                            temp['features_img_gt'] = {str(key): str(value) 
                                                    for key, value in extractor.execute(os.path.join(img_fold, file),'./temp_gt_mask.png',
                                                                            label = 255).items()}
                        except:
                            temp['features_img_gt'] = 'mask not found'
                            
                        try:
                            temp['features_img_gt_not'] = {str(key): str(value) 
                                                        for key, value in extractor.execute(os.path.join(img_fold, file),'./temp_gt_not_mask.png',
                                                                                label = 255).items()}
                        except:
                            temp['features_img_gt_not'] = 'mask not found'
                            
                        try:
                            temp['features_img_pred'] = {str(key): str(value) 
                                                    for key, value in extractor.execute(os.path.join(img_fold, file),'./temp_pred_mask.png',
                                                                            label = 255).items()}
                        except:
                            temp['features_img_pred'] = 'mask not found'
                            
                        try:
                                                
                            # Save the negative segmentation map pred
                            dilated_mask_pred = cv2.dilate(pred, kernel, iterations=1)
                            neg_seg_map_pred = np.logical_not(dilated_mask_pred).astype(np.uint8) * 255

                            neg_seg_map_pred[:border_width, :] = 0
                            neg_seg_map_pred[-border_width:, :] = 0
                            neg_seg_map_pred[:, :border_width] = 0
                            neg_seg_map_pred[:, -border_width:] = 0

                            labeled_array_pred, num_features_pred = label(pred_bw)
                            if labeled_array_pred.size == 0:
                                object_centroid_pred = 0
                            else:
                                coords_pred = np.column_stack(np.where(labeled_array_pred > 0))
                                if coords_pred.size == 0:
                                    object_centroid_pred = 0
                                else:
                                    object_centroid_pred = coords_pred.mean(axis=0)

                            neg_seg_map_pred[:int(object_centroid_pred[0]+offset), :] = 0

                            neg_seg_map_pred_PIL = Image.fromarray(neg_seg_map_pred)
                            neg_seg_map_pred_PIL.save('./temp_pred_not_mask.png')         
                            
                            temp['features_img_pred_not'] = {str(key): str(value) 
                                                        for key, value in extractor.execute(os.path.join(img_fold, file),'./temp_pred_not_mask.png',
                                                                                label = 255).items()}
                        except:
                            temp['features_img_pred_not'] = 'mask not found'
                            
                        summary.append(temp)

                    elif i != 0:
                        file_new = file.replace(class_pred, classes[i])
                        slice_number = int(file_new.split('_')[3].split('.')[0])
                        file_new = file_new.replace(file_new.split('_')[3], str(slice_number+90))
            
                        # Put everything to zero except the current label
                        pred_out_copy = pred_out.copy()
                        pred_out_copy[pred_out_copy != i] = 0

                        pred_bw = pred_out_copy > 0
                        pred_bw_PIL = Image.fromarray(np.uint8(pred_bw*255))
                        pred_bw_PIL.save('./temp_pred_mask.png')

                        # Dilate the object in the 
                        kernel_size = 20
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

                        dilated_mask_pred = cv2.dilate(pred, kernel, iterations=1)
                        
                        # Obtain the negative segmentation map
                        neg_seg_map_pred = np.logical_not(dilated_mask_pred).astype(np.uint8) * 255
                        
                        # Put 15% of pixels to zero on each border
                        border_width = int(0.15 * min(gt.shape[:2]))
                        offset = int(0.10 * min(gt.shape[:2]))
                        
                        neg_seg_map_pred[:border_width, :] = 0
                        neg_seg_map_pred[-border_width:, :] = 0
                        neg_seg_map_pred[:, :border_width] = 0
                        neg_seg_map_pred[:, -border_width:] = 0

                        # Find the centroid of the original object
                        labeled_array_pred, num_features_pred = label(pred_bw)
                        if labeled_array_pred.size == 0:
                            object_centroid_pred = 0
                        else:
                            coords_pred = np.column_stack(np.where(labeled_array_pred > 0))
                            if coords_pred.size == 0:
                                object_centroid_pred = 0
                            else:
                                object_centroid_pred = coords_pred.mean(axis=0)

                        # Put to zero all pixels above the centroid
                        neg_seg_map_pred[:int(object_centroid_pred[0]+offset), :] = 0

                        # Store values
                        
                        temp['File'] = file_new
                        temp['subject'] = file_new.split('_')[0]
                        temp['muscle_code']  = file_new.split('_')[1]
                        temp['side']    = file_new.split('_')[2]

                        temp['class_gt'] = class_gt
                        temp['class_pred'] = class_pred
                        temp['iou'] = np.nan
                        temp['prec'] = np.nan
                        temp['rec'] = np.nan
                        
                        temp['features_img_gt'] = 'mask not found'
                        temp['features_img_gt_not'] = 'mask not found'
                            
                        try:
                            temp['features_img_pred'] = {str(key): str(value) 
                                                    for key, value in extractor.execute(os.path.join(img_fold, file),'./temp_pred_mask.png',
                                                                            label = 255).items()}
                        except:
                            temp['features_img_pred'] = 'mask not found'
                            
                        try:
                            temp['features_img_pred_not'] = {str(key): str(value) 
                                                        for key, value in extractor.execute(os.path.join(img_fold, file),'./temp_pred_not_mask.png',
                                                                                label = 255).items()}
                        except:
                            temp['features_img_pred_not'] = 'mask not found'
                            
                        summary.append(temp)
                else:
                    # file = file
                    
                    gt_bw = gt > 0
                    gt_bw_PIL = Image.fromarray(np.uint8(gt_bw*255))
                    gt_bw_PIL.save('./temp_gt_mask.png')
                    
                    # Dilate the object in the 
                    kernel_size = 20
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

                    dilated_mask = cv2.dilate(gt, kernel, iterations=1)

                    # Obtain the negative segmentation map
                    neg_seg_map = np.logical_not(dilated_mask).astype(np.uint8) * 255
                    
                    # Put 15% of pixels to zero on each border
                    border_width = int(0.15 * min(gt.shape[:2]))
                    offset = int(0.10 * min(gt.shape[:2]))
                    
                    neg_seg_map[:border_width, :] = 0
                    neg_seg_map[-border_width:, :] = 0
                    neg_seg_map[:, :border_width] = 0
                    neg_seg_map[:, -border_width:] = 0

                    # Find the centroid of the original object
                    labeled_array, num_features = label(gt_bw)
                    if labeled_array.size == 0:
                        object_centroid = 0
                    else:
                        coords = np.column_stack(np.where(labeled_array > 0))
                        if coords.size == 0:
                            object_centroid = 0
                        else:
                            object_centroid = coords.mean(axis=0)

                    # Put to zero all pixels above the centroid
                    neg_seg_map[:int(object_centroid[0]+offset), :] = 0
                    
                    # Save the negative segmentation map gt
                    neg_seg_map_PIL = Image.fromarray(neg_seg_map)
                    neg_seg_map_PIL.save('./temp_gt_not_mask.png')
                    
                    # Store values
                    temp['File'] = file
                    temp['subject'] = file.split('_')[0]
                    temp['muscle_code']  = file.split('_')[1]
                    temp['side']    = file.split('_')[2]

                    temp['class_gt'] = class_gt
                    temp['class_pred'] = 'background'
                    temp['iou']  = 0
                    temp['prec'] = 0
                    temp['rec']  = 0
                    
                    try:
                        temp['features_img_gt'] = {str(key): str(value) 
                                                for key, value in extractor.execute(os.path.join(img_fold, file),'./temp_gt_mask.png',
                                                                        label = 255).items()}
                    except:
                        temp['features_img_gt'] = 'mask not found'
                        
                    try:
                        temp['features_img_gt_not'] = {str(key): str(value) 
                                                    for key, value in extractor.execute(os.path.join(img_fold, file),'./temp_gt_not_mask.png',
                                                                            label = 255).items()}
                    except:
                        temp['features_img_gt_not'] = 'mask not found'
                        
                    temp['features_img_pred'] = 'mask not found'
                    temp['features_img_pred_not'] = 'mask not found'
                        
                    summary.append(temp)

        
        fold = fold + 1
            
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

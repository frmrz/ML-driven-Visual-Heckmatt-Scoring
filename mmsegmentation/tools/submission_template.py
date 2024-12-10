import os
import numpy as np
import torch
import json

from mmseg.apis import init_model, inference_model

from PIL import Image
from tqdm import tqdm

from mmengine import Config

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

from skimage import morphology, measure

working_folder = os.path.dirname(os.path.abspath(__file__))

# Define group name
group_name = 'Usopp'

checkpoint_path = '/home/francesco/Desktop/POLI/RADBOUD/CODE/work_dir/VERSE/iter_40000.pth'
cfg_path = '/home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/VERSE_experiments/VERSE_config.py'
test_img_folder = '/home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/data/VERSE/img_dir/test'

results_folder = f'/home/francesco/Desktop/POLI/DIDATTICA/EIM/24-25/Contest/RESULTS/{group_name}'

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Ground truth masks folder
gt_mask_folder = '/home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/data/VERSE/ann_dir/test'

# Classes and palette for VERSE dataset
classes = ('background', 'object')
paletteVERSE = [
    (0, 0, 0),       # background - black
    (255, 255, 255)  # object - white
]

@DATASETS.register_module()
class VERSE(BaseSegDataset):
    METAINFO = dict(classes=classes, palette=paletteVERSE)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png',
                         seg_map_suffix='.png',
                         reduce_zero_label=False,
                         **kwargs)

cfg = Config.fromfile(cfg_path)

# Initialize the model from the config and the checkpoint
model = init_model(cfg, checkpoint_path, 'cuda:0')

# Get image list and divide the dataset into subjects
img_list = os.listdir(test_img_folder)

# Extract subject names
subjects = []
for img_name in img_list:
    subject = img_name.split('_')[0]
    if subject not in subjects:
        subjects.append(subject)

# Create a dictionary with subject names as keys and image lists as values
subject_dict = {}
for subject in subjects:
    imgs = [img for img in os.listdir(test_img_folder) if img.startswith(subject)]
    imgs.sort()
    subject_dict[subject] = imgs

###############################
#### Function Definitions #####
###############################

def create_3d_volume(folder, img_list):
    """Create a 3D volume from a list of image filenames in a folder."""
    volume = []
    for img_name in img_list:
        img_path = os.path.join(folder, img_name)
        if os.path.isfile(img_path) and img_name.endswith('.png'):
            img = Image.open(img_path)
            img = np.array(img)
            # Convert to binary if necessary
            if img.max() > 1:
                img = img / 255
            volume.append(img)
    volume = np.stack(volume, axis=0)>0
    return volume

def calculate_3d_dice(pred_volume, gt_volume):
    """Calculate the 3D Dice coefficient between two volumes."""
    pred_flat = pred_volume.flatten()
    gt_flat = gt_volume.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    dice = (2. * intersection) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-7)
    return dice

def count_connected_objects(volume):
    """Count the number of connected components in a binary volume."""
    labeled_volume = measure.label(volume, connectivity=1)
    num_objects = np.max(labeled_volume)
    return num_objects

def calculate_absolute_error(pred_count, gt_count):
    """Calculate the absolute error in counting connected objects."""
    error = abs(gt_count - pred_count)
    return error

###############################
####### Inference Loop ########
###############################

# Loop over the subjects and their images
for subject, img_list in subject_dict.items():
    # Create a folder for each subject in the results folder
    subject_folder = os.path.join(results_folder, subject)
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    for img_name in tqdm(img_list, desc=f'Processing {subject}'):
        if img_name.endswith('.png'):
            img_path = os.path.join(test_img_folder, img_name)

            ###############################
            ##### 2D Pre-Processing #######
            ###############################

            img = Image.open(img_path)
            img = np.array(img)

            # Normalize the image intensity to [0,1]
            img = img / 255.0

            # Convert back to PIL Image if necessary for the model
            img = Image.fromarray((img * 255).astype(np.uint8))

            # Save img to temporary file
            img.save('tmp_img.png')

            ###############################
            #######    Inference   ########
            ###############################

            result = inference_model(model, 'tmp_img.png')

            # Get data from the result
            pred_label = result.pred_sem_seg.data.squeeze()
            pred_label = pred_label.cpu().numpy().astype(np.uint8)

            ###############################
            ###### 2D Post-Processing #####
            ###############################

            # Remove small objects using skimage
            pred_label = morphology.remove_small_objects(pred_label.astype(bool), min_size=3, connectivity=1)

            # Apply morphological closing to fill small holes
            pred_label = morphology.binary_closing(pred_label, morphology.disk(3))

            # Convert pred_label to uint8
            pred_label = pred_label.astype(np.uint8)

            # pred_label to 0-255
            pred_label = pred_label * 255

            ###############################
            #######   Save results  #######
            ###############################

            # Save the result
            pred_label_img = Image.fromarray(pred_label)
            pred_label_img.save(os.path.join(subject_folder, img_name))

###############################
####### Evaluation Loop #######
###############################

# Initialize a dictionary to store evaluation results
evaluation_results = {}

for subject in subjects:
    
    subject_folder = os.path.join(results_folder, subject)
    pred_img_list = [img for img in os.listdir(subject_folder) if img.endswith('.png')]
    pred_img_list.sort()

    # Create predicted volume
    pred_volume = create_3d_volume(subject_folder, pred_img_list)

    # Remove small objects in 3D
    pred_volume = pred_volume.astype(np.uint8)

    # Count connected objects in predicted volume
    pred_num_objects = count_connected_objects(pred_volume)

    # Create ground truth volume
    gt_img_list = [img for img in os.listdir(gt_mask_folder) if img.startswith(subject)]
    gt_img_list.sort()

    # Ensure that the predicted and ground truth image lists are aligned
    if pred_img_list != gt_img_list:
        print(f"Warning: Image lists for subject {subject} do not match between prediction and ground truth.")
        # Adjust lists if necessary

    # Create ground truth volume
    gt_volume = create_3d_volume(gt_mask_folder, gt_img_list)

    # Remove small holes, separate the close objects and remove small objects
    gt_volume = morphology.binary_opening(gt_volume, morphology.ball(3))
    gt_volume = morphology.remove_small_objects(gt_volume.astype(bool), min_size=25)
    gt_volume = gt_volume.astype(np.uint8)
    
    # Convert gt_volume to binary
    gt_volume = (gt_volume > 0).astype(np.uint8)

    # Calculate 3D Dice coefficient
    dice_score = calculate_3d_dice(pred_volume, gt_volume)

    # Count connected objects in ground truth volume
    gt_num_objects = count_connected_objects(gt_volume)

    # Calculate absolute error in counting
    count_error = calculate_absolute_error(pred_num_objects, gt_num_objects)

    # Store the evaluation results
    evaluation_results[subject] = {
        'dice_score': f"{dice_score:.3f}",
        'absolute_count_error': str(count_error)
    }

# Save the evaluation results to a JSON file
evaluation_json_path = os.path.join(results_folder, f'{group_name}.json')
with open(evaluation_json_path, 'w') as json_file:
    json.dump(evaluation_results, json_file, indent=4)

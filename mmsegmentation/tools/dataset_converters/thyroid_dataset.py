# script to prepare thyroid data for analysis with mmsegmentation

import os
import PIL.Image as Image
from sklearn.model_selection import train_test_split

data_folder = '/media/francesco/DEV001/PROJECT-THYROID/DATA'
destination_folder = '/media/francesco/DEV001/PROJECT-THYROID/CODE/mmsegmentation/data/thyroid_A'

# Let's begin with dataset A

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

image_folder = os.path.join(data_folder, 'A_256', 'IMAGES')
mask_folder = os.path.join(data_folder, 'A_256', 'LABELS')

# get all images names
images = os.listdir(image_folder)
images = [os.path.join(image_folder, image) for image in images]

# get all masks names
masks = os.listdir(mask_folder)
masks = [os.path.join(mask_folder, mask) for mask in masks]

# split data into train, val and test
images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.1, random_state=42)
images_train, images_val, masks_train, masks_val = train_test_split(images_train, masks_train, test_size=0.1, random_state=42)

subsets = ['train', 'val', 'test']

for subset in subsets:

    # create subset folder
    subset_folder_images = os.path.join(destination_folder, 'img_dir', subset)
    subset_folder_masks = os.path.join(destination_folder, 'ann_dir', subset)

    if not os.path.exists(subset_folder_images):
        os.makedirs(subset_folder_images)
    if not os.path.exists(subset_folder_masks):
        os.makedirs(subset_folder_masks)

    if subset == 'train':
        images = images_train
        masks = masks_train 
    elif subset == 'val':
        images = images_val
        masks = masks_val
    else:
        images = images_test
        masks = masks_test

    for image, mask in zip(images, masks):
        # copy images
        image_name = os.path.basename(image)
        destination_image = os.path.join(subset_folder_images, image_name)
        os.system('cp {} {}'.format(image, destination_image))
        # copy masks
        mask_name = os.path.basename(mask)
        destination_mask = os.path.join(subset_folder_masks, mask_name)
        os.system('cp {} {}'.format(mask, destination_mask))

# Let's now prepare dataset B

destination_folder = '/media/francesco/DEV001/PROJECT-THYROID/CODE/mmsegmentation/data/thyroid_B'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

image_folder = os.path.join(data_folder, 'B_256', 'IMAGES')
mask_folder = os.path.join(data_folder, 'B_256', 'LABELS')

# get all images names
images = os.listdir(image_folder)
images = [os.path.join(image_folder, image) for image in images]

# get all masks names
masks = os.listdir(mask_folder)
masks = [os.path.join(mask_folder, mask) for mask in masks]

# split data into train, val and test
images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.1, random_state=42)
images_train, images_val, masks_train, masks_val = train_test_split(images_train, masks_train, test_size=0.1, random_state=42)

subsets = ['train', 'val', 'test']

for subset in subsets:
    # create subset folder
    subset_folder_images = os.path.join(destination_folder, 'img_dir', subset)
    subset_folder_masks = os.path.join(destination_folder, 'ann_dir', subset)

    if not os.path.exists(subset_folder_images):
        os.makedirs(subset_folder_images)
    if not os.path.exists(subset_folder_masks):
        os.makedirs(subset_folder_masks)

    if subset == 'train':
        images = images_train
        masks = masks_train 
    elif subset == 'val':
        images = images_val
        masks = masks_val
    else:
        images = images_test
        masks = masks_test

    for image, mask in zip(images, masks):
        # copy images
        image_name = os.path.basename(image)
        destination_image = os.path.join(subset_folder_images, image_name)
        os.system('cp {} {}'.format(image, destination_image))
        # copy masks
        mask_name = os.path.basename(mask)
        destination_mask = os.path.join(subset_folder_masks, mask_name)
        os.system('cp {} {}'.format(mask, destination_mask))

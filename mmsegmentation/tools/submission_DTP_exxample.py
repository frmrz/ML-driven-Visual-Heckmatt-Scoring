import os
import numpy as np
import torch

from mmseg.apis import init_model, inference_model
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from mmengine import Config

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

from skimage import morphology


working_folder = os.path.dirname(os.path.abspath(__file__))
print(f'Working folder: {working_folder}')

# checkpoint_path = os.path.join(working_folder, 'checkpoints', 'dtp_r50-d8_512x512_40k_voc12aug_20200606_161938-1b9f5c0e.pth')
# cfg_path = os.path.join(working_folder, 'configs', 'dtp', 'dtp_r50-d8_512x512_40k_voc12aug.py')

checkpoint_path = '/media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/DTP/iter_40000.pth'
cfg_path = '/media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/DTP/DTP_config.py'

# test_img_folder = os.path.join(working_folder, 'test_images')
# print(f'Test images folder: {test_img_folder}')

test_img_folder = '/media/francesco/DEV001/DIDATTICA/EIM/CONTEST-DTP/TEST/manual'

# results_folder = os.path.join(working_folder, 'results')
# if not os.path.exists(results_folder):
#     os.makedirs(results_folder)

results_folder =  '/media/francesco/DEV001/PROJECT-THYROID/RESULTS/Usopp'

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# classes DTP
classes = ('background',
        'cell')

paletteDTP = [
    (0, 0, 0), # background - black
    (255, 255, 255), # cell - white
]

@DATASETS.register_module()
class DTP(BaseSegDataset):
    METAINFO = dict(classes = classes, palette = paletteDTP)
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png',
                        seg_map_suffix='.png',
                        reduce_zero_label = False,
                        **kwargs)
        
cfg = Config.fromfile(cfg_path)
print(f'Config:\n{cfg.pretty_text}')

# Init the model from the config and the checkpoint
model = init_model(cfg, checkpoint_path, 'cuda:0')


# Loop over the test images

for img_name in tqdm(os.listdir(test_img_folder)):

    if img_name.endswith('.png'):
        img_path = os.path.join(test_img_folder, img_name)

        ###############################
        ####### Pre-Processing ########
        ###############################
        
        img = Image.open(img_path)
        img = np.array(img)

        # convert to 0-1 range and then to 0-255 (uint8)
        # img = img / 65536 * 255
        # img = img.astype(np.uint8)

        # save img to temporary file
        img = Image.fromarray(img)
        img.save('tmp_img.png')

        ###############################
        #######    Inference   ########
        ###############################

        result = inference_model(model, 'tmp_img.png')

        # get data from the result
        pred_label = result.pred_sem_seg.data.squeeze()
        pred_label = pred_label.cpu().numpy().astype(np.uint8)

        ###############################
        ####### Post-Processing #######
        ###############################

        # remove small objects using skimage
        pred_label = morphology.remove_small_objects(pred_label, min_size=3, connectivity=1)

        # # dilate the image using skimage    
        # pred_label = morphology.dilation(pred_label, morphology.square(3))

        # # erode the image using skimage
        # pred_label = morphology.erosion(pred_label, morphology.square(3))

        # pred_label to 0-255
        pred_label = pred_label * 255
        pred_label = pred_label.astype(np.uint8)

        ###############################
        #######   Save results  #######
        ###############################

        # save the result
        pred_label = Image.fromarray(pred_label)
        pred_label.save(os.path.join(results_folder, img_name))

    
    
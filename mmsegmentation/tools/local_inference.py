" Script for inference and data visualization using mmsegmentation trained models"

import os
import numpy as np
import torch

from mmseg.apis import init_model, inference_model
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from mmengine import Config
import argparse

from torchmetrics.classification import MulticlassJaccardIndex
from skimage import color

import json
import re

def extract_parameters(filename):

    # Read the file content
    with open(filename, 'r') as file:
        file_content = file.read()

    # Regular expressions to find 'num_classes' and 'ignore_index'
    num_classes_pattern = r"num_classes\s*=\s*(\d+)"
    ignore_index_pattern = r"ignore_index\s*=\s*(\d+|-?\d+)"

    # Search for 'num_classes'
    num_classes_match = re.search(num_classes_pattern, file_content)
    num_classes = int(num_classes_match.group(1)) if num_classes_match else None

    # Search for 'ignore_index'
    ignore_index_match = re.search(ignore_index_pattern, file_content)
    ignore_index = int(ignore_index_match.group(1)) if ignore_index_match else None

    return num_classes, ignore_index

def parse_args():
    parser = argparse.ArgumentParser(description='Inference a model')
    parser.add_argument('config', help='test config file path')

    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--img_folder', help='folder with images to be segmented')
    parser.add_argument('--out_results', help='folder to save results')

    parser.add_argument('--tta', action='store_true', help='Test time augmentation')

    parser.add_argument('--ground_truth', help='folder with ground truth images')
    parser.add_argument('--plot_rgb', action='store_true', help='Plot labels in fake RGB')
    parser.add_argument('--plot_label_compare', action='store_true', help='Plot figure with image-gt_label-pred_label')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.test_dataloader['batch_size'] = 1

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # construct the model and load checkpoint
    model = init_model(cfg, args.checkpoint, device='cuda:0')

    # test a single image
    img_folder = args.img_folder
    out_folder = os.path.join(args.out_results)
    pred_out_folder = os.path.join(out_folder, 'pred')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if not os.path.exists(pred_out_folder):
        os.makedirs(pred_out_folder)
        
    # search for the string "num_classes=" in the config file and get the number of classes
    # this is a hacky way to get the number of classes
    num_classes, ignore_index = extract_parameters(args.config)
        
    if args.ground_truth:
        metric = MulticlassJaccardIndex(num_classes=num_classes,
                                        ignore_index=ignore_index,
                                        average='macro')
        ious = []
    
    if args.plot_rgb:

        cmap = plt.cm.get_cmap('hsv', num_classes)

            
        # Create an array of colors from the color map
        palette = (cmap(np.arange(num_classes))[:, :3] * 255).astype(int)

        rgb_out_folder = os.path.join(out_folder, 'rgb_labels')
        
        if not os.path.exists(rgb_out_folder):
            os.makedirs(rgb_out_folder)
    
    if args.plot_label_compare:
        label_compare_out_folder = os.path.join(out_folder, 'label_compare')
        
        if not os.path.exists(label_compare_out_folder):
            os.makedirs(label_compare_out_folder)


    for img_name in tqdm(os.listdir(img_folder)):

        img_path = os.path.join(img_folder, img_name)
        result = inference_model(model, img_path)

        # get data from the result
        pred_label = result.pred_sem_seg.data.squeeze()
        pred_label = pred_label.cpu().numpy().astype(np.uint8)

        # save the visualization results to image files
        out_path = os.path.join(pred_out_folder, img_name)
        Image.fromarray(pred_label).save(out_path)

        if args.ground_truth:
            gt_path = os.path.join(args.ground_truth, img_name)
            if gt_path.endswith('.png'):
                gt_label = Image.open(gt_path)
                gt_label = np.array(gt_label)
            elif gt_path.endswith('.jpg'):
                gt_label = Image.open(gt_path.replace('.jpg', '.png'))
                gt_label = np.array(gt_label)


            # # if the gt label and the pred label have different shapes, center crop the pred label to match the gt label
            # if gt_label.shape != pred_label.shape:
            #     # get the delta between the rows and columns
            #     delta_rows = pred_label.shape[0] - gt_label.shape[0]
            #     delta_cols = pred_label.shape[1] - gt_label.shape[1]

            #     # get the number of rows and columns to be cropped and round them tho get the indices
            #     crop_rows = delta_rows // 2
            #     crop_cols = delta_cols // 2

            #     # crop the pred label
            #     pred_label = pred_label[crop_rows:crop_rows + gt_label.shape[0],
            #                             crop_cols:crop_cols + gt_label.shape[1]]

            # # if one size of gt_label is smaller than 256, pad it to 256
            # if gt_label.shape[0] < 256:
            #     pad_rows = 256 - gt_label.shape[0]
            #     gt_label = np.pad(gt_label, ((0, pad_rows), (0, 0)), 'constant', constant_values=0)

            # if gt_label.shape[1] < 256:
            #     pad_cols = 256 - gt_label.shape[1]
            #     gt_label = np.pad(gt_label, ((0, 0), (0, pad_cols)), 'constant', constant_values=0)

            temp = dict()
            temp[img_name] = metric(torch.from_numpy(pred_label), torch.from_numpy(gt_label)).numpy()
            ious.append(temp)

        if args.plot_rgb:
            cmap = plt.cm.get_cmap('hsv', num_classes)

            # Create an array of colors from the color map
            palette = (cmap(np.arange(num_classes))[:, :3] * 255).astype(int)

            rgb_out_folder = os.path.join(out_folder, 'rgb_labels')

            # make an rgb map based on number of classes using skimage.color.label2rgb
            rgb_label = Image.fromarray(color.label2rgb(pred_label, colors=palette).astype(np.uint8))
            out_path = os.path.join(rgb_out_folder, img_name)
            rgb_label.save(out_path)

        if args.plot_label_compare:
            gt_path = os.path.join(args.ground_truth, img_name)
            if gt_path.endswith('.png'):
                gt_label = Image.open(gt_path)
                gt_label = np.array(gt_label)
            elif gt_path.endswith('.jpg'):
                gt_label = Image.open(gt_path.replace('.jpg', '.png'))
                gt_label = np.array(gt_label)

            # # if the gt label and the pred label have different shapes, center crop the pred label to match the gt label
            # if gt_label.shape != pred_label.shape:
            #     # get the delta between the rows and columns
            #     delta_rows = pred_label.shape[0] - gt_label.shape[0]
            #     delta_cols = pred_label.shape[1] - gt_label.shape[1]

            #     # get the number of rows and columns to be cropped and round them tho get the indices
            #     crop_rows = delta_rows // 2
            #     crop_cols = delta_cols // 2

            #     # crop the pred label
            #     pred_label = pred_label[crop_rows:crop_rows + gt_label.shape[0],
            #                             crop_cols:crop_cols + gt_label.shape[1]]
                
            rgb_label_gt = Image.fromarray(color.label2rgb(gt_label, colors=palette).astype(np.uint8))
            rgb_label_pred = Image.fromarray(color.label2rgb(pred_label, colors=palette).astype(np.uint8))
            rgb_img = Image.open(img_path).convert('RGB')

            # rgb_img = np.array(rgb_img)

            # # if the image and the gt label have different shapes, center crop the image to match the gt label
            # if rgb_img.shape != gt_label.shape:
            #     # get the delta between the rows and columns
            #     delta_rows = rgb_img.shape[0] - gt_label.shape[0]
            #     delta_cols = rgb_img.shape[1] - gt_label.shape[1]

            #     # get the number of rows and columns to be cropped and round them tho get the indices
            #     crop_rows = delta_rows // 2
            #     crop_cols = delta_cols // 2

            #     # crop the pred label
            #     rgb_img = rgb_img[crop_rows:crop_rows + gt_label.shape[0],
            #                             crop_cols:crop_cols + gt_label.shape[1]]

            label_compare = np.hstack((np.array(rgb_img), np.array(rgb_label_gt), np.array(rgb_label_pred)))
            out_path = os.path.join(label_compare_out_folder, img_name)
            Image.fromarray(label_compare).save(out_path)

    if args.ground_truth:
        # print mean and std of ious
        ious_values = [item for sublist in ious for item in sublist.values()]
        print('Mean iou: ', np.mean(ious_values))
        print('Std iou: ', np.std(ious_values))
        
        # convert ndarrays to string
        ious = [{key: str(value) for key, value in individual_iou.items()} for individual_iou in ious]

        # save ious to file json
        with open(os.path.join(out_folder, 'ious.json'), 'w') as f:
            json.dump(ious, f)


if __name__ == '__main__':
    main()

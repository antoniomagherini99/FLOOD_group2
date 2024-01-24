# .py file containing the functions used for data augmentation in ConvLSTM model for DSAIE FLOOD project.

from PIL import Image
import matplotlib.pyplot as plt
import random
import pandas as pd
import os

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F

class MultiFixedRotation:
    def __init__(self, fixed_angles):
        self.angles = fixed_angles

    def __call__(self):
        angle = random.choice(self.angles)
        return angle

# List of fixed rotation angles (in degrees)

fixed_angles = [0, 90, 180, 270]

def augmentation(train_dataset, p_hflip=0.5, p_vflip=0.5, range_t=len(train_dataset)): #angles=fixed_angles, 
    '''
    Function for implementing data augmentation of inputs (DEM, X- and Y-Slope, 
    Water Depth and Discharge).

    Input: train_dataset = torch tensor, dataset with input variables
           p_hflip, p_vflip = float, probability of horizontal and vertical flipping
                              default = 0.5 for both
           angles = angle degrees for dataset rotation, fixed at 0°, 90°, 180°, 270°
    Output:  
    '''
    # implement transformation pipeline with horizontal and vertical flip and rotation of fixed angles
    transformation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=p_hflip),
    transforms.RandomVerticalFlip(p=p_vflip)]) 
    #transforms.functional.rotate(train_dataset[i] for i in range(len(train_dataset)), RandomFixedRotation(angles))

    # transform dataset
    transformed_dataset = [transformation_pipeline(train_dataset) for _ in range(range_t)]
    #plot([orig_img] + transformed_dataset)
    return transformed_dataset

def plot(dataset, row_title=None, **imshow_kwargs):
    """Plooting function taken from https://raw.githubusercontent.com/pytorch/vision/main/gallery/transforms/helpers.py"""
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(dataset)
    num_cols = len(dataset[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
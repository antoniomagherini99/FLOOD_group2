# .py file containing the functions used for data augmentation in ConvLSTM model for DSAIE FLOOD project.

from PIL import Image
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
# from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F

# List of fixed rotation angles (in degrees)
fixed_angles = [0, 90, 180, 270]

class MultiFixedRotation:
    '''
    Class that implements random rotations of the dataset at fixed angles
    '''
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

def augmentation(train_dataset, angles=[90,180,270], p_hflip=0.5, full=True):
    '''
    Function for implementing data augmentation of inputs (DEM, X- and Y-Slope,
    Water Depth, and Discharge).

    Input: train_dataset = torch tensor, dataset with input variables
           angles = list of angle degrees for random rotation of the dataset, 
                    default values are 90°, 180°, 270°
           p_hflip = float, probability of horizontal flipping
                     default = 0.5 
           
    Output: transformed_dataset = new dataset with augmented data,
                                  if full = True returns the original and trasformed dataset concatenated together
                                  if full = False returns only the trasformed dataset 
    '''
    
    # transformation pipeline with horizontal flip
    transformation_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=p_hflip)])
    
    # rotation with MultiFixedRotation class
    multi_fixed_rotation = MultiFixedRotation(angles)
    
    # initialize lists needed for looping
    inputs = []
    outputs = []

    transformed_inputs = []
    transformed_outputs = []

    for idx in range(len(train_dataset)):
        inputs.append(train_dataset[idx][0])
        outputs.append(train_dataset[idx][1]) 
        
        # apply augmentation (flipping + rotation)
        transformed_inputs.append(multi_fixed_rotation(transformation_pipeline(train_dataset[idx][0])))
        transformed_outputs.append(multi_fixed_rotation(transformation_pipeline(train_dataset[idx][1])))   

    # stack lists
    inputs_stack = torch.stack(inputs)
    outputs_stack = torch.stack(outputs)

    transformed_inputs = [torch.tensor(arr) for arr in transformed_inputs]
    transformed_outputs = [torch.tensor(arr) for arr in transformed_outputs]

    transformed_inputs = torch.stack(transformed_inputs)
    transformed_outputs = torch.stack(transformed_outputs)

    # concatenate tensors
    all_inputs = torch.cat([inputs_stack, transformed_inputs])
    all_outputs = torch.cat([outputs_stack, transformed_outputs])

    transformed_dataset = torch.utils.data.TensorDataset(all_inputs, all_outputs)
    
    if full==True:
        print(f'The samples in the dataset before augmentation were {len(train_dataset)}\n\
The samples in the dataset after augmentation are {len(transformed_dataset)}')
    
    if full==False:
        transformed_dataset = torch.utils.data.TensorDataset(transformed_inputs, transformed_outputs)
        warning_msg = (
f'\nBe careful, you are not including the transformed dataset into the original one!\n\
The samples in the dataset before augmentation were {len(train_dataset)}\n\
The samples in the dataset after augmentation are {len(transformed_dataset)}\n\
You are now using only the transformed dataset.'
)
        warnings.warn(warning_msg, RuntimeWarning)
    return transformed_dataset

    
# def plot(dataset, row_title=None, **imshow_kwargs):
#     """Plooting function taken from https://raw.githubusercontent.com/pytorch/vision/main/gallery/transforms/helpers.py"""
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]

#     num_rows = len(dataset)
#     num_cols = len(dataset[0])
#     _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         for col_idx, img in enumerate(row):
#             boxes = None
#             masks = None
#             if isinstance(img, tuple):
#                 img, target = img
#                 if isinstance(target, dict):
#                     boxes = target.get("boxes")
#                     masks = target.get("masks")
#                 elif isinstance(target, tv_tensors.BoundingBoxes):
#                     boxes = target
#                 else:
#                     raise ValueError(f"Unexpected target type: {type(target)}")
#             img = F.to_image(img)
#             if img.dtype.is_floating_point and img.min() < 0:
#                 # Poor man's re-normalization for the colors to be OK-ish. This
#                 # is useful for images coming out of Normalize()
#                 img -= img.min()
#                 img /= img.max()

#             img = F.to_dtype(img, torch.uint8, scale=True)
#             if boxes is not None:
#                 img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
#             if masks is not None:
#                 img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

#             ax = axs[row_idx, col_idx]
#             ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     if row_title is not None:
#         for row_idx in range(num_rows):
#             axs[row_idx, 0].set(ylabel=row_title[row_idx])

#     plt.tight_layout()
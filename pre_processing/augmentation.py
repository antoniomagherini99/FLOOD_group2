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

class MultiFixedRotation:
    def __init__(self, fixed_angles):
        self.angles = fixed_angles

    def __call__(self):
        angle = random.choice(self.angles)
        return angle

# List of fixed rotation angles (in degrees)

fixed_angles = [0, 90, 180, 270]

def augmentation(train_dataset, range_t, p_hflip=0.5, p_vflip=0.5, full=True):
    '''
    Function for implementing data augmentation of inputs (DEM, X- and Y-Slope,
    Water Depth, and Discharge).

    Input: train_dataset = torch tensor, dataset with input variables
           p_hflip, p_vflip = float, probability of horizontal and vertical flipping
                              default = 0.5 for both
           angles = angle degrees for dataset rotation, fixed at 0°, 90°, 180°, 270°
    Output:
    '''
    
    # Define the transformation pipeline with horizontal and vertical flip
    transformation_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=p_hflip),
        transforms.RandomVerticalFlip(p=p_vflip)])
     #transforms.functional.rotate(train_dataset[i] for i in range(len(train_dataset)), RandomFixedRotation(angles))
    
    inputs = []
    outputs = []

    transformed_inputs = []
    transformed_outputs = []

    for idx in range(len(train_dataset)):
        inputs.append(train_dataset[idx][0])
        outputs.append(train_dataset[idx][1]) 
        
        transformed_inputs.append(transformation_pipeline(train_dataset[idx][0]))
        transformed_outputs.append(transformation_pipeline(train_dataset[idx][1]))   

    # Assuming transformed_inputs and transformed_outputs are lists of tensors
    inputs_stack = torch.stack(inputs)
    outputs_stack = torch.stack(outputs)

    transformed_inputs = [torch.tensor(arr) for arr in transformed_inputs]
    transformed_outputs = [torch.tensor(arr) for arr in transformed_outputs]

    # Now, use torch.stack to concatenate the tensors along a new dimension
    transformed_inputs = torch.stack(transformed_inputs)
    transformed_outputs = torch.stack(transformed_outputs)

    all_inputs = torch.cat([inputs_stack, transformed_inputs])
    all_outputs = torch.cat([outputs_stack, transformed_outputs])

    # Now, create TensorDataset
    transformed_dataset = torch.utils.data.TensorDataset(all_inputs, all_outputs)
    print(f'The samples in the dataset before augmentation were {len(train_dataset)}\n\
The samples in the dataset after augmentation are {len(transformed_dataset)}')
    
    if full==False:
        transformed_dataset = torch.utils.data.TensorDataset(transformed_inputs, transformed_outputs)
        print(f'Be careful! You are not including the transformed dataset into the original one!\n\
The samples in the dataset before augmentation were {len(train_dataset)}\n\
The samples in the dataset after augmentation are {len(transformed_dataset)} ')
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
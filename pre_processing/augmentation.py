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
        # random.seed(seed)
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

# def augmentation(train_dataset, angles=[90,180,270], p_hflip=0.5, full=True):
#     '''
#     Function for implementing data augmentation of inputs (DEM, X- and Y-Slope,
#     Water Depth, and Discharge).

#     Input: train_dataset = torch tensor, dataset with input variables
#            seed = int, number for keeping the same random choice for trasforming both inputs and outputs in the same way, 
#                   default = 42 
#            angles = list of angle degrees for random rotation of the dataset, 
#                     default = 90°, 180°, 270°
#            p_hflip = float, probability of horizontal flipping
#                      default = 0.5 
           
#     Output: transformed_dataset = new dataset with augmented data,
#                                   if full = True returns the original and trasformed dataset concatenated together
#                                   if full = False returns only the trasformed dataset 
#     '''
    
#     # transformation pipeline with horizontal flip
#     transformation_pipeline = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=p_hflip)]) #transforms.RandomVerticalFlip(p=p_hflip)
    
#     inputs_sizes = train_dataset[0][0].shape
#     outputs_sizes = train_dataset[0][1].shape
#     # rotation with MultiFixedRotation class
#     # fixed_rotation = MultiFixedRotation(angles)
    
#     # initialize lists needed for looping
#     inputs = []
#     outputs = []

#     transformed_inputs = []
#     transformed_outputs = []

#     for idx in range(len(train_dataset)):
#         inputs.append(train_dataset[idx][0])
#         outputs.append(train_dataset[idx][1]) 
        
#         # apply augmentation (flipping + rotation)
#         # transformed_inputs.append(transformation_pipeline(train_dataset[idx][0])) # fixed_rotation(..., seed)
#         # transformed_outputs.append(transformation_pipeline(train_dataset[idx][1]))   

#     inputs_flattened = inputs.flatten(1,1)
#     outputs_flattened = outputs.flatten(1,2)
#     print(np.shape(inputs_flattened))
#     print(np.shape(inputs_flattened))

#     # transformed_dataset = multi_fixed_rotation(transformation_pipeline(train_dataset))

#     # stack lists
#     inputs_stack = torch.stack(inputs)
#     outputs_stack = torch.stack(outputs)

#     transformed_inputs = [torch.tensor(i) for i in transformed_inputs]
#     transformed_outputs = [torch.tensor(i) for i in transformed_outputs]

#     transformed_inputs = torch.stack(transformed_inputs)
#     transformed_outputs = torch.stack(transformed_outputs)

#     # concatenate tensors
#     all_inputs = torch.cat([inputs_stack, transformed_inputs])
#     all_outputs = torch.cat([outputs_stack, transformed_outputs])

#     transformed_dataset = torch.utils.data.TensorDataset(all_inputs, all_outputs)
    
#     if full==True:
#         print(f'The samples in the dataset before augmentation were {len(train_dataset)}\n\
# The samples in the dataset after augmentation are {len(transformed_dataset)}')
    
#     if full==False:
#         transformed_dataset = torch.utils.data.TensorDataset(transformed_inputs, transformed_outputs)
#         warning_msg = (
# f'\nBe careful, you are not including the transformed dataset into the original one!\n\
# The samples in the dataset before augmentation were {len(train_dataset)}\n\
# The samples in the dataset after augmentation are {len(transformed_dataset)}\n\
# You are now using only the transformed dataset.'
# )
#         warnings.warn(warning_msg, RuntimeWarning)
#     return transformed_dataset

def augmentation(train_dataset, angles=[90,180,270], p_hflip=0.5, full=True):
    '''
    Function for implementing data augmentation of inputs (DEM, X- and Y-Slope,
    Water Depth, and Discharge).

    Input: train_dataset = torch tensor, dataset with input variables
           seed = int, number for keeping the same random choice for trasforming both inputs and outputs in the same way, 
                  default = 42 
           angles = list of angle degrees for random rotation of the dataset, 
                    default = 90°, 180°, 270°
           p_hflip = float, probability of horizontal flipping
                     default = 0.5 
           
    Output: transformed_dataset = new dataset with augmented data,
                                  if full = True returns the original and trasformed dataset concatenated together
                                  if full = False returns only the trasformed dataset 
    '''
    
    # transformation pipeline with horizontal flip
    transformation_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=p_hflip)])  
        # transforms.RandomVerticalFlip(p=p_hflip)
    
    # rotation with MultiFixedRotation class
    fixed_rotation = MultiFixedRotation(angles)

    n_samples = len(train_dataset)
    
    # trasformed_dataset = fixed_rotation(transformation_pipeline(train_dataset)) 

    # initialize lists needed for looping   
    inputs = []
    outputs = []

    transformed_inputs = []
    transformed_outputs = []

    for idx in range(len(train_dataset)):
        
        inputs.append(train_dataset[idx][0])
        outputs.append(train_dataset[idx][1])
        
        # seed = random.randint(0, 100)
        # # apply augmentation (flipping + rotation)
        # transformed_inputs.append(transformation_pipeline(train_dataset[idx][0])) # fixed_rotation(..., seed)
        # transformed_outputs.append(transformation_pipeline(train_dataset[idx][1]))   

        # transformed_inputs.append(trasformed_dataset[idx][0])
        # transformed_outputs.append(trasformed_dataset[idx][1])

    # get sizes of each dimension
    inputs_sizes = train_dataset[0][0].shape
    outputs_sizes = train_dataset[0][1].shape
    # inputs_sizes = [len(inputs[0]), len(inputs[0][0]), len(inputs[0][0][0]), len(inputs[0][0][0][0])]
    # outputs_sizes = [len(outputs[0]), len(outputs[0][0]), len(outputs[0][0][0]), len(outputs[0][0][0][0])]
    print(f'Inputs sizes: {inputs_sizes},\n\
Outputs sizes: {outputs_sizes}\n')

    # stack lists
    inputs_tensor = torch.stack(inputs)
    outputs_tensor = torch.stack(outputs)

    # dimension of inputs channels over which concatentate 
    inputs_flat_dim = inputs_tensor.size()[2]
    # print('flat_dim', inputs_flat_dim) 

    flattened_inputs = inputs_tensor.flatten(0,1) #alternatively try .view(-1, 64, 64) 
    flattened_outputs = outputs_tensor.flatten(1,2)
    concat = torch.cat([flattened_inputs, flattened_outputs], dim=1)

    transformed_concat = fixed_rotation(transformation_pipeline(concat))
    # print(f'Transformed concat: {transformed_concat.shape}')

    # reshape the tensors to original dimensions
    transformed_inputs = transformed_concat[:, :inputs_flat_dim, :, :].view(n_samples, inputs_sizes[0], 
                                                                            inputs_sizes[1], inputs_sizes[2], inputs_sizes[3]) 
    transformed_outputs = transformed_concat[:, inputs_flat_dim:, :, :].view(n_samples, outputs_sizes[0], 
                                                                            outputs_sizes[1], outputs_sizes[2], outputs_sizes[3])

    # print(f'Transformed inputs: {transformed_inputs.shape}')
    # print(f'Transformed inputs: {transformed_outputs.shape}')
    
    # trasformed_inputs = transformed_concat[]

    # transformed_inputs = [torch.tensor(i) for i in transformed_inputs]
    # transformed_outputs = [torch.tensor(i) for i in transformed_outputs]

    # transformed_inputs = torch.stack(transformed_inputs)
    # transformed_outputs = torch.stack(transformed_outputs)

    # concatenate tensors
    all_inputs = torch.cat([inputs_tensor, transformed_inputs])
    all_outputs = torch.cat([outputs_tensor, transformed_outputs])

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
# .py file containing the functions used for data augmentation in ConvLSTM model for DSAIE FLOOD project.

import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F

# List of fixed rotation angles (in degrees)
fixed_angles = [0, 90, 180, 270]

class MultiFixedRotation:
    '''
    Class for making random rotations of the dataset at fixed angles. If a list is given as input, the function randmoly choses
    one single angle every time it is called.

    Input: angles = single value or list of angles from which make a random choice
           x = tensor to be rotated.

    Output: rotated_dataset = tensor, dataset after rotation with random angle 
    '''
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        rotated_dataset = transforms.functional.rotate(x, angle) 
        return rotated_dataset
    
def rotate(train_dataset, angle):
    '''
    Function to implement fixed rotation at specified angles
    '''
    rotated_dataset = transforms.functional.rotate(train_dataset, angle)
    return rotated_dataset

def augmentation(train_dataset, angles=[90,180,270], p_hflip=0.5, full=True):
    '''
    Function for implementing data augmentation of the whole dataset (both inputs and outputs).

    Input: train_dataset = torch tensor, dataset with inputs and targets
           angles = list of angle degrees for random rotation of the dataset,
                    default = 90°, 180°, 270° for keeping the characteristics of the dataset (boundary conditions, dimension etc.)
           p_hflip = float, probability of horizontal flipping of the dataset,
                     default = 0.5 
           full = key, to choose whether to return a new dataset concatenating the original and transformed one or not,
                  default = True, the function returns the full new dataset   
           
    Output: transformed_dataset = new dataset with augmented data,
                                  if full = True returns the original and trasformed dataset concatenated together
                                  if full = False returns only the trasformed dataset 
    '''
    
    # transformation pipeline with horizontal flip
    transformation_pipeline = transforms.Compose([transforms.RandomHorizontalFlip(p=p_hflip)])  
    
    # rotation with MultiFixedRotation class
    # fixed_rotation = MultiFixedRotation(angles)

    n_samples = len(train_dataset)

    # initialize lists needed for looping   
    inputs = []
    targets = []

    for idx in range(len(train_dataset)):
        inputs.append(train_dataset[idx][0])
        targets.append(train_dataset[idx][1])

    # get sizes of inputs and targets
    inputs_sizes = train_dataset[0][0].shape
    targets_sizes = train_dataset[0][1].shape

    # stack lists to get tensors
    inputs_tensor = torch.stack(inputs)
    targets_tensor = torch.stack(targets)

    # get dimension of inputs channels over which concatentate 
    flat_dim = int(inputs_sizes[1])

    # flatten the tensors before concatenating 
    flattened_inputs = inputs_tensor.flatten(0,1) 
    flattened_targets = targets_tensor.flatten(1,2)
    
    # concatenate flattened tensors over 2nd dimension
    concat = torch.cat([flattened_inputs, flattened_targets], dim=1)

    # transform original dataset - flip 
    flipped_dataset = transformation_pipeline(concat)

    # transform original dataset - only rotation with different angles
    rotated_concat1 = rotate(concat, 90)
    rotated_concat2 = rotate(concat, 180)
    rotated_concat3 = rotate(concat, 270)

    # transform original dataset - combine flipping and rotation with different angles
    rotated_flipped1 = rotate(flipped_dataset, 90)
    rotated_flipped2 = rotate(flipped_dataset, 180)
    rotated_flipped3 = rotate(flipped_dataset, 270)

    # concatenate all transformed datasets
    list_to_concat = [rotated_concat1, rotated_concat2, rotated_concat3,
                              rotated_flipped1, rotated_flipped2, rotated_flipped3]
    final_concat = torch.cat(list_to_concat)

    # get length of the new dataset length for reshaping the full transformed dataset 
    len_concat = len(list_to_concat)

    # reshape the tensors to original dimensions
    transformed_inputs = final_concat[:, :flat_dim, :, :].view(n_samples*len_concat, inputs_sizes[0], 
                                                                           inputs_sizes[1], inputs_sizes[2], inputs_sizes[3]) 
    transformed_targets = final_concat[:, flat_dim:, :, :].view(n_samples*len_concat, targets_sizes[0], 
                                                                           targets_sizes[1], targets_sizes[2], targets_sizes[3])

    # concatenate tensors
    all_inputs = torch.cat([inputs_tensor, transformed_inputs])
    all_outputs = torch.cat([targets_tensor, transformed_targets])

    # create Dataset type
    transformed_dataset = torch.utils.data.TensorDataset(all_inputs, all_outputs)
    
    if full==True:
        print(f'The samples in the dataset before augmentation were {len(train_dataset)}\n\
The samples in the dataset after augmentation are {len(transformed_dataset)}')
    
    if full==False:
        transformed_dataset = torch.utils.data.TensorDataset(transformed_inputs, transformed_targets)
        warning_msg = (
f'\nBe careful, you are not including the transformed dataset into the original one!\n\
The samples in the dataset before augmentation were {len(train_dataset)}\n\
The samples in the dataset after augmentation are {len(transformed_dataset)}\n\
You are now using only the transformed dataset.'
)
        warnings.warn(warning_msg, RuntimeWarning)
    return transformed_dataset
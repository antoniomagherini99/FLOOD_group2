# .py file containing the functions used for encoding and decoding the datasets for faster uploading it in ConvLSTM model for DSAIE FLOOD project.

import pandas as pd
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset

from pre_processing.load_datasets import retrieve_path, count_pixels

def encode_into_csv(inputs, targets, train_val_test):
    """
    Due to the long run time of computing all inputs and targets,
    these will be encoded into a csv file to reduce the computational duration

    Parameters
    ----------
    inputs: torch.tensor of shape: (samples x input_features = 3 x pixel x pixel)
        which represents the inputs. Contains DEM, and slopes x & y.
    targets: torch.tensor of shape: (samples x time steps x target_features x 
        pixel x pixel) which represents the targets. Contains all time steps 
        including time step 0 and half time stpes.
    train_val_test: key for specifying what we are using the model for
        'train_val' = train and validate the model
        'test1' = test the model with dataset 1
        'test2' = test the model with dataset 2
        'test3' = test the model with dataset 3

    Returns
    -------
    df_inputs: pandas.dataframe
        Contains the flatened array of inputs
        length = (num_samples x input_features = 3 x pixel x pixel)
    df_targets: pandas.dataframe
        Contains the flattened array of targets
        length = (num_samples x time_steps x target_features = 2 x pixel x pixel)
    """
    # Flatten the tensors and concatenate them along the specified dimension
    flattened_tensor1 = torch.flatten(inputs, start_dim=0)
    flattened_tensor2 = torch.flatten(targets, start_dim=0)

    # Convert the tensor to a pandas DataFrame
    df_inputs = pd.DataFrame(flattened_tensor1.numpy())
    df_targets = pd.DataFrame(flattened_tensor2.numpy())

    dir_path = retrieve_path(train_val_test)
    
    # Save the DataFrame to a CSV file
    df_inputs.to_csv(dir_path + train_val_test + '_in.csv', index=False)

    # if train_val_test = 'train_val' targets file is too big to be loaded in GitHub
    # and it needs to be split into 4 different .csv files
    # n_tot = 63569920 total number of rows of targets (80x2x97x64x64)
    # n = n_tot/4 to split in 4 separate files
    n_tot = int(targets.size(0) * targets.size(1) * targets.size(2) * targets.size(3) * targets.size(4))
    n = int(n_tot / 4)
    
    # Definetely a way to automate this with a for loop
    if train_val_test in ('train_val', 'test2', 'test3'):
        df_targets[:n].to_csv(dir_path + train_val_test + '_tar1.csv', index=False)
        df_targets[n:2*n].to_csv(dir_path + train_val_test + '_tar2.csv', index=False)
        df_targets[2*n:3*n].to_csv(dir_path + train_val_test + '_tar3.csv', index=False)
        df_targets[3*n:].to_csv(dir_path + train_val_test + '_tar4.csv', index=False)
    else: 
        df_targets.to_csv(dir_path + train_val_test + '_tar.csv', index=False)
    return df_inputs, df_targets

def decode_from_csv(train_val_test):
    """
    Due to the long run time of computing all inputs and targets, a
    .csv file will be opened at the start of every notebook which
    represents the inputs and targets for a certain dataset.

    Parameters
    ----------
    train_val_test: key for specifying what we are using the model for
        'train_val' = train and validate the model
        'test1' = test the model with dataset 1
        'test2' = test the model with dataset 2
        'test3' = test the model with dataset 3

    Returns
    -------
    dataset: contains two torch.Tensors with length equal to number of samples
        Shape: (num_samples x 2)
        index [:, 0]: inputs. Contains DEM, slope x and y and the boundary
            condition for water depth for all files in a dataset.
            Shape: (time_step = 1 x num_features = 4 x pixel x pixel)
        index [:, 1]: targets. Contains water depth and discharge for all files
            in a dataset. Shape: (time_steps x num_features = 2 x pixel x pixel)
            
    """
    dir_path = retrieve_path(train_val_test)
    
    df_inputs = pd.read_csv(dir_path + train_val_test + '_in.csv')
    
    # if train_val_test = 'train_val', targets file is too big to be loaded in GitHub
    # and it needs to be split into 4 different .csv files and then concatenated together
    if train_val_test in ('train_val', 'test2', 'test3'):
        df_targets1 = pd.read_csv(dir_path + train_val_test + '_tar1.csv')
        df_targets2 = pd.read_csv(dir_path + train_val_test + '_tar2.csv')
        df_targets3 = pd.read_csv(dir_path + train_val_test + '_tar3.csv')
        df_targets4 = pd.read_csv(dir_path + train_val_test + '_tar4.csv')

        df_targets = pd.concat([df_targets1, df_targets2, 
                                df_targets3, df_targets4], axis=0) 
    else: # test 1 is small enough
        df_targets = pd.read_csv(dir_path + train_val_test + '_tar.csv')

    # Convert the DataFrame to a PyTorch tensor
    restored_inputs = torch.tensor(df_inputs.values)
    restored_targets = torch.tensor(df_targets.values)

    # Use of WD, as all want to know how many time steps are in dataset
    dir_path = retrieve_path(train_val_test) + 'WD/'
    count = len(os.listdir(dir_path))
       
    # used to count timesteps
    wd_file = dir_path + str(os.listdir(dir_path)[0]) # first file in folder
    time_steps = np.loadtxt(wd_file).shape[0]
    
    pixel_square = count_pixels(train_val_test)
    
    #calculate number of fetures in inputs and targets given length of dataset:
    input_features = int(len(restored_inputs) / (count * pixel_square ** 2))
    target_features = int(len(restored_targets) / (count * time_steps * pixel_square ** 2))
    
    shape_inputs = (count, input_features, pixel_square, pixel_square)
    shape_targets = (count, time_steps, target_features, pixel_square, pixel_square)

    # Split the restored tensor into two tensors based on the original shapes
    inputs = torch.reshape(restored_inputs, shape_inputs)
    targets = torch.reshape(restored_targets, shape_targets)
    
    inputs = inputs.unsqueeze(1) # demonstrate that inputs has 1 time step
    
    boundary = targets[:, 0, 0].unsqueeze(1).unsqueeze(1) # only interested in water depth, discharge is zeros
    targets = targets[:, 2::2] # remove boundaries from targets and halve the time steps
    
    inputs = torch.cat((inputs, boundary), dim = 2)
    
    dataset = TensorDataset(inputs.float(), targets.float())
    
    # Print the shapes of the restored tensors
    print("Restored inputs Shape:", inputs.shape)
    print("Restored targets Shape:", targets.shape)
    return dataset
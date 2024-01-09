import pandas as pd
import torch
import os
import re

from load_datasets import retrieve_path, count_pixels

def encode_into_csv(inputs, targets, train_val_test):
    """
    Due to the long run time of computing all inputs and targets, these will be encoded into a csv file
    to reduce the computatio duration

    Input:
    inputs: torch.tensor of shape: samples x 3 x 64 x 64 which represents the inputs of the network
    targets: torch.tensor of shape: samples x time steps x 64 x 64 which represents the targets of the network
    train_val_test: str, differentiate between csv files

    Outputs:
    None: But a csv file is create with a predetermined name
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
    Due to the long run time of computing all inputs and targets, a .csv file will be opened
    at the start of every notebook which represents the inputs and targets for a certain dataset.

    Input:
    train_val_test: str, identifies which dataset is being retrieved

    Output:
    inputs: torch.Tensor which contains DEM, slope x and y for all files in a dataset
            Shape is samples x 3 x pixel_square
    targets: torch.Tensor which contains water depth and discharge for all files in a dataset.
            Shape is samples x 2 x time_steps x pixel_square x pixel_square
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
    else:
        df_targets = pd.read_csv(dir_path + train_val_test + '_tar.csv')

    # Convert the DataFrame to a PyTorch tensor
    restored_inputs = torch.tensor(df_inputs.values)
    restored_targets = torch.tensor(df_targets.values)

    # Automatic way to know how many samples are in a given dataset folder
    count = 0
    dir_path = retrieve_path(train_val_test) + 'DEM/' # Arbitrary choice as DEM, VX, VY and WD all have the same number of samples
    for path in os.listdir(dir_path):
        if count == 0:
            file_number = re.search(r'\d{1,5}', path)
            pixel_square = count_pixels(int(file_number.group()), train_val_test)
        else:
            None
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
        else:
            None

    shape_inputs = (count, 3, pixel_square, pixel_square)
    shape_targets = (count, 97, 2, pixel_square, pixel_square)

    # Split the restored tensor into two tensors based on the original shapes
    inputs = torch.reshape(restored_inputs, shape_inputs)
    targets = torch.reshape(restored_targets, shape_targets)
    
    targets = targets.permute(0, 2, 1, 3, 4) # to have a similar shape as inputs

    # Print the shapes of the restored tensors
    print("Restored inputs Shape:", inputs.shape)
    print("Restored targets Shape:", targets.shape)
    return inputs, targets
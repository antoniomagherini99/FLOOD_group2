# file for storing functions used for loading datasets
# 1st version - Antonio

import torch 
import os

import numpy as np

from numba import jit, prange, njit

# The following paths access the main folder (i.e., dataset_train_val, dataset1 and so on). 
# The path of the specific type of data (DEM, VX and so on) is to be specified after.
path_train = f'../dataset_train_val/' 
path_test1 = f'../dataset1/'
path_test2 = f'../dataset2/'
path_test3 = f'../dataset3/'

# ------------- #

# The following lines create variables to more easily specify what we use the model for 
# (i.e., train and validate, test with dataset 1 and so on) in the following functions.

train_val = 'train_val'
test1 = 'test1'
test2 = 'test2'
test3 = 'test3'

# ------------- #

# @njit
def process_elevation_data(file_id, train_val_test='train_val'):
    """
    Processes elevation data from a DEM file.

    Input:
    file_id (str): Identifier of the DEM file to be processed.
    train_val_test: key for specifying what we are using the model for
                   'train_val' = train and validate the model
                   'test1' = test the model with dataset 1
                   'test2' = test the model with dataset 2
                   'test3' = test the model with dataset 3

    Output:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """
    # specify what we use the model for -- so far works for only one specified input (i.e., file_id), 
    # will need to be improved to work for all inputs regardless of the number of the file
    if train_val_test == 'train_val':
        file_path = path_train + f'DEM/DEM_{file_id}.txt'
    elif train_val_test == 'test1':
        file_path = path_test1 + f'DEM/DEM_{file_id}.txt'
    elif train_val_test == 'test2':
        file_path = path_test2 + f'DEM/DEM_{file_id}.txt'
    elif train_val_test == 'test3':
        file_path = path_test3 + f'DEM/DEM_{file_id}.txt'

    # # Construct the file path from the given file identifier
    # file_path = f'DEM_{file_id}.txt'

    # Load the elevation data from the file
    elevation_data = np.loadtxt(file_path)

    # Reshape the elevation data into a 64x64 grid
    elevation_grid = elevation_data[:, 2].reshape(64, 64)

    # Convert the elevation grid to a PyTorch tensor
    elevation_tensor = torch.tensor(elevation_grid)

    # Compute the slope in the x and y directions
    slope_x, slope_y = torch.gradient(elevation_tensor)

    # Combine the elevation tensor with the slope tensors
    elevation_slope_tensor = torch.stack((elevation_tensor, slope_x, slope_y), dim=0)

    return elevation_slope_tensor

# ------------- #

# @njit
def process_water_depth(file_id, train_val_test='train_val', time_step=0):
    """
    Processes water depth data from a specific time step in a file.

    Args:
    file_id (str): Identifier of the water depth file to be processed.
    train_val_test: key for specifying what we are using the model for
                   'train_val' = train and validate the model
                   'test1' = test the model with dataset 1
                   'test2' = test the model with dataset 2
                   'test3' = test the model with dataset 3
    time_step (int): Time step to extract from the file. Default is the first time step. Default is the first time step.

    Returns:
    torch.Tensor or None: A 64x64 tensor representing water depth at the given time step, or None if the data is invalid.
    """
    # specify what we use the model for -- so far works for only one specified input (i.e., file_id), 
    # will need to be improved to work for all inputs regardless of the number of the file
    if train_val_test == 'train_val':
        file_path = path_train + f'WD/WD_{file_id}.txt'
    elif train_val_test == 'test1':
        file_path = path_test1 + f'WD/WD_{file_id}.txt'
    elif train_val_test == 'test2':
        file_path = path_test2 + f'WD/WD_{file_id}.txt'
    elif train_val_test == 'test3':
        file_path = path_test3 + f'WD/WD_{file_id}.txt'

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    try:
        # Extract the specified row and convert string elements to floats
        selected_row = lines[time_step].split()
        depth_values = [float(val) for val in selected_row]

        # Validate and reshape the data into a 64x64 tensor
        if len(depth_values) == 64 * 64:
            depth_tensor = torch.tensor(depth_values).view(64, 64)
            return depth_tensor
        else:
            raise ValueError(f"The number of elements in {file_path} at time step {time_step} doesn't match a 64x64 matrix.")
    except IndexError:
        raise IndexError(f"Time step {time_step} is out of range for the file {file_path}.")

# ------------- #

# @njit
def process_velocities(file_id, train_val_test='train_val', time_step=0):
    """
    Processes elevation data from a DEM file.

    Input:
    file_id (str): Identifier of the DEM file to be processed.
    train_val_test: key for specifying what we are using the model for
                   'train_val' = train and validate the model
                   'test1' = test the model with dataset 1
                   'test2' = test the model with dataset 2
                   'test3' = test the model with dataset 3
    time_step (int): Time step to extract from the file. Default is the first time step. Default is the first time step.

    Output:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """
    # specify what we use the model for -- so far works for only one specified input (i.e., file_id), 
    # will need to be improved to work for all inputs regardless of the number of the file
    if train_val_test == 'train_val':
        file_path_x = path_train + f'VX/VX_{file_id}.txt'
        file_path_y = path_train + f'VY/VY_{file_id}.txt'
    
    elif train_val_test == 'test1':
        file_path_x = path_train + f'VX/VX_{file_id}.txt'
        file_path_y = path_train + f'VY/VY_{file_id}.txt'
    
    elif train_val_test == 'test2':
        file_path_x = path_train + f'VX/VX_{file_id}.txt'
        file_path_y = path_train + f'VY/VY_{file_id}.txt'
    
    elif train_val_test == 'test3':
        file_path_x = path_train + f'VX/VX_{file_id}.txt'
        file_path_y = path_train + f'VY/VY_{file_id}.txt'

    # Load the elevation data from the file
    vx, vy = np.loadtxt(file_path_x), np.loadtxt(file_path_y)

    # Read the file
    with open(file_path_x, 'r') as file:
        lines_x = file.readlines()
    
    with open(file_path_y, 'r') as file:
        lines_y = file.readlines()

    try:
        # Extract the specified row and convert string elements to floats
        selected_row_x = lines_x[time_step].split()
        selected_row_y = lines_y[time_step].split()
        
        vel_x = [float(val) for val in selected_row_x]
        vel_y = [float(val) for val in selected_row_y]

        # Validate and reshape the data into a 64x64 tensor
        if (len(vel_x) == 64 * 64) and (len(vel_y) == 64 * 64):
            vel_x = torch.tensor(vel_x).view(64, 64)
            vel_y = torch.tensor(vel_y).view(64, 64)
            return vel_x, vel_y
        else:
            raise ValueError(f"The number of elements in {file_path_x} or {file_path_y} at time step {time_step} doesn't match a 64x64 matrix.")
    except IndexError:
        raise IndexError(f"Time step {time_step} is out of range for the file {file_path_x} or {file_path_y}.")

# ------------- #

# @njit  
def compute_targets(file_id, train_val_test = 'train_val', time_step = 0):
    """
    Use process_velocities and process_water_depth to compute discharge

    Input:
    file_id (str): Identifier of the DEM file to be processed.
    train_val_test: key for specifying what we are using the model for
                   'train_val' = train and validate the model
                   'test1' = test the model with dataset 1
                   'test2' = test the model with dataset 2
                   'test3' = test the model with dataset 3

    Output:
    targets: A torch.tensor which is 2 x 64 x 64. Both targets are water depth and discharge respectively
    """
    water_depth = process_water_depth(file_id, train_val_test, time_step)
    vx, vy = process_velocities(file_id, train_val_test, time_step)

    magnitude = torch.sqrt(vx**2 + vy**2)
    discharge = water_depth * magnitude # per meter width

    targets = torch.stack((water_depth, discharge), dim=0)
    return targets

# ------------- #

# @jit(nopython=False, parallel=True)
def load_all_boys(train_val_test, time=97):
    '''
    Load all "file_id" and "time_step" for chosen dataset

    Input: 
    train_val_test = key for choosing dataset  
         = 'train_val', 'test1', 'test2', 'test3'
    time = time step of simulation # 97 is hardcoded !

    Output:
    inputs: torch.Tensor which contains DEM, slope x and y for all files in a dataset
            Shape is samples x 3 x 64 x 64
    targets: torch.Tensor which contains water depth and discharge for all files in a dataset.
             Shape is samples x time steps x 2 x 64 x 64
    '''
    if train_val_test == 'train_val':
        file_path = path_train
    elif train_val_test == 'test1':
        file_path = path_test1
    elif train_val_test == 'test2':
        file_path = path_test2
    elif train_val_test == 'test3':
        file_path = path_test3
    
    count = 0
    dir_path = file_path + 'DEM' # Arbitrary choice as DEM, vx, vy and WD all have the same number of samples
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    inputs = torch.zeros((count, 3, 64, 64))
    targets = torch.zeros((count, time, 2, 64, 64))

    for i in range(count):
        print(i)
        inputs[i] = process_elevation_data(i + 1, train_val_test)
        for t in range(time):
            targets[i, t] = compute_targets(i + 1, train_val_test, time_step = t)
    return inputs, targets
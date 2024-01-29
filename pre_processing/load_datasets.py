# file for storing functions used for loading datasets

import torch 
import os
import re

import numpy as np

def retrieve_path(train_val_test):
    """
    Retrieve the path of a given data folder using a predefined dictionary
    
    Parameters
    ----------
    train_val_test : str
        Identifier of dictionary. Expects:
            'train_val', 'test1', 'test2', 'test3'.

    Returns
    -------
    path : str
        path of datafolder.

    """
    path_dictionary = {
        'train_val': 'data/dataset_train_val/',
        'test1': 'data/dataset1/',
        'test2': 'data/dataset2/',
        'test3': 'data/dataset3/'
        }
    path = path_dictionary[train_val_test]
    return path

# ------------- #

def count_pixels(train_val_test):
    """
    Calculate the number of pixels contained in row/column of image of a
    given dataset

    Parameters
    ----------
    train_val_test : str
        key for specifying what we are using the model for
            'train_val' = train and validate the model
            'test1' = test the model with dataset 1
            'test2' = test the model with dataset 2
            'test3' = test the model with dataset 3

    Returns
    -------
    pixel_square : int
        Number of pixels in a row/column of the image.

    """
    dir_path = retrieve_path(train_val_test)
    
    folder_path = dir_path + 'DEM/' # Arbitrary choice as DEM, vx, vy and WD all have the same number of samples
    
    file_path = folder_path + str(os.listdir(folder_path)[0]) # first file in folder
    
    elevation_data = np.loadtxt(file_path)

    tot_pixels = len(elevation_data[:, 2])
    pixel_square = int(np.sqrt(tot_pixels)) # Image is always a square
    return pixel_square

# @njit
def process_elevation_data(file_id, train_val_test='train_val', pixel_square = 64):
    """
    Processes elevation data from a DEM file.

    Input:
    file_id (int): Identifier of the DEM file to be processed.
    train_val_test: key for specifying what we are using the model for
                   'train_val' = train and validate the model
                   'test1' = test the model with dataset 1
                   'test2' = test the model with dataset 2
                   'test3' = test the model with dataset 3
    pixel_square : int
        Number of pixels in a row/column of the image.

    Output:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """
    # specify what we use the model for -- so far works for only one specified input (i.e., file_id), 
    # will need to be improved to work for all inputs regardless of the number of the file
    dir_path = retrieve_path(train_val_test)
    file_path = dir_path + f'DEM/DEM_{file_id}.txt'

    # # Construct the file path from the given file identifier
    # file_path = f'DEM_{file_id}.txt'

    # Load the elevation data from the file
    elevation_data = np.loadtxt(file_path)

    # Reshape the elevation data into a square grid
    elevation_grid = elevation_data[:, 2].reshape(pixel_square, pixel_square)

    # Convert the elevation grid to a PyTorch tensor
    elevation_tensor = torch.tensor(elevation_grid)

    # Compute the slope in the x and y directions
    slope_x, slope_y = torch.gradient(elevation_tensor)

    # Combine the elevation tensor with the slope tensors
    elevation_slope_tensor = torch.stack((elevation_tensor, slope_x, slope_y), dim=0)

    return elevation_slope_tensor

# ------------- #

def process_water_depth(file_id, train_val_test='train_val', time_step=0, pixel_square = 64):
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
    pixel_square : int
        Number of pixels in a row/column of the image.

    Returns:
    torch.Tensor or None: A 64x64 tensor representing water depth at the given time step, or None if the data is invalid.
    """
    # specify what we use the model for -- so far works for only one specified input (i.e., file_id), 
    # will need to be improved to work for all inputs regardless of the number of the file
    dir_path = retrieve_path(train_val_test)
    file_path = dir_path + f'WD/WD_{file_id}.txt'

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    try:
        # Extract the specified row and convert string elements to floats
        selected_row = lines[time_step].split()
        depth_values = [float(val) for val in selected_row]

        # Validate and reshape the data into a square tensor
        if len(depth_values) == pixel_square * pixel_square:
            depth_tensor = torch.tensor(depth_values).view(pixel_square, pixel_square)
            return depth_tensor
        else:
            raise ValueError(f"The number of elements in {file_path} at time step {time_step} doesn't match a 64x64 matrix.")
    except IndexError:
        raise IndexError(f"Time step {time_step} is out of range for the file {file_path}.")

# ------------- #

def process_velocities(file_id, train_val_test='train_val', time_step=0, pixel_square = 64):
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
    pixel_square : int
        Number of pixels in a row/column of the image.

    Output:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """
    # specify what we use the model for -- so far works for only one specified input (i.e., file_id), 
    # will need to be improved to work for all inputs regardless of the number of the file
    dir_path = retrieve_path(train_val_test)
    file_path_x = dir_path + f'VX/VX_{file_id}.txt'
    file_path_y = dir_path + f'VY/VY_{file_id}.txt'

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
        if (len(vel_x) == pixel_square * pixel_square) and (len(vel_y) == pixel_square * pixel_square):
            vel_x = torch.tensor(vel_x).view(pixel_square, pixel_square)
            vel_y = torch.tensor(vel_y).view(pixel_square, pixel_square)
            return vel_x, vel_y
        else:
            raise ValueError(f"The number of elements in {file_path_x} or {file_path_y} at time step {time_step} doesn't match a 64x64 matrix.")
    except IndexError:
        raise IndexError(f"Time step {time_step} is out of range for the file {file_path_x} or {file_path_y}.")

# ------------- #

def compute_targets(file_id, train_val_test = 'train_val', time_step = 0, pixel_square = 64):
    """
    Use process_velocities and process_water_depth to compute discharge

    Input:
    file_id (str): Identifier of the DEM file to be processed.
    train_val_test: key for specifying what we are using the model for
                   'train_val' = train and validate the model
                   'test1' = test the model with dataset 1
                   'test2' = test the model with dataset 2
                   'test3' = test the model with dataset 3
    pixel_square : int
        Number of pixels in a row/column of the image.

    Output:
    targets: A torch.tensor which is 2 x 64 x 64. Both targets are water depth and discharge respectively
    """
    water_depth = process_water_depth(file_id, train_val_test, time_step, pixel_square)
    vx, vy = process_velocities(file_id, train_val_test, time_step, pixel_square)

    magnitude = torch.sqrt(vx**2 + vy**2)
    discharge = water_depth * magnitude # per meter width

    targets = torch.stack((water_depth, discharge), dim=0)
    return targets

# ------------- #

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
    file_path = retrieve_path(train_val_test)
    
    count = 0
    dir_path = file_path + 'DEM' # Arbitrary choice as DEM, vx, vy and WD all have the same number of samples
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
        else:
            None
    
    pixel_square = count_pixels(train_val_test)
    inputs = torch.zeros((count, 3, pixel_square, pixel_square))
    targets = torch.zeros((count, time, 2, pixel_square, pixel_square))

    i = 0
    for path in os.listdir(dir_path):
        file_number = re.search(r'\d{1,5}', path)
        print(file_number)
        print(int(file_number.group()))
        inputs[i] = process_elevation_data(int(file_number.group()),
                                           train_val_test, pixel_square)
        for t in range(time):
            targets[i, t] = compute_targets(int(file_number.group()),
                                            train_val_test, time_step = t,
                                            pixel_square = pixel_square)
        i += 1
    return inputs, targets
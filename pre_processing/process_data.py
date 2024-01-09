import torch
import numpy as np

def process_elevation_data(file_id, dataset_id):
    """
    Processes elevation data from a DEM file.

    Args:
    file_id (str): Identifier of the DEM file to be processed.
    dataset_id (str): Identifier of which dataset is to be used (train/val, dataset1, dataset2 or dataset3).

    Returns:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """
    try:
    # retrieve file_path
        if dataset_id=='train/val':
            file_path = f'data/dataset_train_val/DEM/DEM_{file_id}.txt'
        elif dataset_id=='dataset1':
            file_path = f'data/dataset1/DEM/DEM_{file_id}.txt'
        elif dataset_id=='dataset2':
            file_path = f'data/dataset2/DEM/DEM_{file_id}.txt'
        elif dataset_id=='dataset3':
            file_path = f'data/dataset3/DEM/DEM_{file_id}.txt'
        else:
            print('invalid dataset identifier entered. Please select one of the following:')
            print('train/val, dataset1, dataset2 or dataset3')

        elevation_data = np.loadtxt(file_path)
        elevation_grid = elevation_data[:, 2].reshape(64, 64)
        elevation_tensor = torch.tensor(elevation_grid, dtype=torch.float32)
        slope_x, slope_y = torch.gradient(elevation_tensor)
        elevation_slope_tensor = torch.stack((elevation_tensor, slope_x, slope_y), dim=0)

        return elevation_slope_tensor

    except Exception as e:
        print(f"Error processing elevation data: {e}")
        return None

    
    
def process_water_depth(file_id, dataset_id, time_step=0):
    """
    Processes water depth data from a specific time step in a file.

    Args:
    file_id (str): Identifier of the water depth file to be processed.
    dataset_id (str): Identifier of which dataset is to be used (train/val, dataset1, dataset2 or dataset3).
    time_step (int): Time step to extract from the file. Default is the first time step.

    Returns:
    torch.Tensor or None: A 64x64 tensor representing water depth at the given time step, or None if the data is invalid.
    """  
    # retrieve file_path
    if dataset_id=='train/val':
        file_path = f'data/dataset_train_val/WD/WD_{file_id}.txt'
    elif dataset_id=='dataset1':
        file_path = f'data/dataset1/WD/WD_{file_id}.txt'
    elif dataset_id=='dataset2':
        file_path = f'data/dataset2/WD/WD_{file_id}.txt'
    elif dataset_id=='dataset3':
        file_path = f'data/dataset3/WD/WD_{file_id}.txt'
    else:
        print('invalid dataset identifier entered. Please select one of the following:')
        print('train/val, dataset1, dataset2 or dataset3')

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
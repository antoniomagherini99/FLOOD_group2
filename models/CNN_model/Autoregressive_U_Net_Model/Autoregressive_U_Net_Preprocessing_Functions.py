import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize
import os
import imageio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import random_split
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib.colors import TwoSlopeNorm
from IPython.display import Image, display

# Normalization

def normalize_sample(sample, scaler_x, scaler_y):
    x = sample['inputs']
    y = sample['targets']
    norm_x = (x - scaler_x.min_[0]) / (scaler_x.data_max_[0] - scaler_x.data_min_[0])
    norm_y = (y - scaler_y.min_[0]) / (scaler_y.data_max_[0] - scaler_y.data_min_[0])
    return {"inputs": norm_x, "targets": norm_y}

# Training and validation dataset preprocessing

def process_elevation_data_for_training(file_id):
    """
    Processes elevation data from a DEM file.

    Args:
    file_id (str): Identifier of the DEM file to be processed.

    Returns:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """

    # Construct the file path from the given file identifier
    file_path = f'data/dataset_train_val/DEM/DEM_{file_id}.txt'

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

def process_water_depth_for_training(file_id, time_step=0):
    """
    Processes water depth data from a specific time step in a file.

    Args:
    file_id (str): Identifier of the water depth file to be processed.
    time_step (int): Time step to extract from the file. Default is the first time step. Default is the first time step.

    Returns:
    torch.Tensor or None: A 64x64 tensor representing water depth at the given time step, or None if the data is invalid.
    """
    file_path = f'data/dataset_train_val/WD/WD_{file_id}.txt'

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

# Test Dataset 1 Preprocessing

def process_elevation_data_for_testing_dataset_1(file_id):
    """
    Processes elevation data from a DEM file.

    Args:
    file_id (str): Identifier of the DEM file to be processed.

    Returns:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """

    # Construct the file path from the given file identifier
    file_path = f'data/dataset1/DEM/DEM_{file_id}.txt'

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

def process_water_depth_for_testing_dataset_1(file_id, time_step=0):
    """
    Processes water depth data from a specific time step in a file.

    Args:
    file_id (str): Identifier of the water depth file to be processed.
    time_step (int): Time step to extract from the file. Default is the first time step. Default is the first time step.

    Returns:
    torch.Tensor or None: A 64x64 tensor representing water depth at the given time step, or None if the data is invalid.
    """
    file_path = f'data/dataset1/WD/WD_{file_id}.txt'

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
        
# Test Dataset 2 Preprocessing
        
def process_elevation_data_for_testing_dataset_2(file_id):
    """
    Processes elevation data from a DEM file.

    Args:
    file_id (str): Identifier of the DEM file to be processed.

    Returns:
    torch.Tensor: A tensor combining the original elevation data and its slope in x and y directions.
    """

    # Construct the file path from the given file identifier
    file_path = f'data/dataset2/DEM/DEM_{file_id}.txt'

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

def process_water_depth_for_testing_dataset_2(file_id, time_step=0):
    """
    Processes water depth data from a specific time step in a file.

    Args:
    file_id (str): Identifier of the water depth file to be processed.
    time_step (int): Time step to extract from the file. Default is the first time step. Default is the first time step.

    Returns:
    torch.Tensor or None: A 64x64 tensor representing water depth at the given time step, or None if the data is invalid.
    """
    file_path = f'data/dataset2/WD/WD_{file_id}.txt'

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
        
# The code below creates a GIF of a dataset

# Create plot/GIF of an example dataset

def plot_and_save_tensors(tensors, file_number, titles, filename):
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors, figsize=(12, 5))

    if num_tensors == 1:
        axes = [axes]

    for ax, tensor, title in zip(axes, tensors, titles):
        # Define 'cmap' and 'vmin', 'vmax' for each type of plot
        if title.startswith('Elevation'):
            cmap = 'terrain'
            vmin, vmax = None, None  # or set specific min/max values for elevation
        elif 'Water Depth' in title:
            cmap = 'Blues'
            vmin, vmax = 0, 1  # Fixed scale for water depth
        else:
            cmap = 'coolwarm'
            vmin, vmax = None, None  # or set specific min/max values for other types of plots

        # Use 'imshow' with the specified 'cmap', 'vmin', and 'vmax'
        im = ax.imshow(tensor, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

        fig.colorbar(im, ax=ax)
        ax.set_title(title)

    plt.suptitle(f'Dataset ID: {file_number}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)

def create_water_depth_gif(file_id, start_time_step, end_time_step, fps=10, gif_filename='water_depth_animation.gif', loop=0):
    images = []

    for WD_time_step in range(start_time_step, end_time_step + 1):
        elevation_slope_tensor = process_elevation_data_for_training(file_id)
        water_depth_tensor = process_water_depth_for_training(file_id, time_step=WD_time_step)
        combined_tensors = list(elevation_slope_tensor) + [water_depth_tensor]

        water_depth_title = f'Water Depth after\n{WD_time_step*30} min'
        titles = ['Elevation', 'Slope X', 'Slope Y', water_depth_title]

        # Filename for the saved image
        filename = f"plot_{WD_time_step}.png"
        plot_and_save_tensors(combined_tensors, file_id, titles, filename)

        images.append(imageio.imread(filename))

    # Save the GIF with the specified loop parameter
    imageio.mimsave(gif_filename, images, fps=fps, loop=loop)

    # Display the GIF in the notebook
    display(Image(filename=gif_filename))
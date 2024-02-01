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

def plot_water_depth_comparison(DEM, real_WD, pred_WD):
    """
    Plots comparison between real and predicted water depth along with DEM and difference.

    Args:
    DEM (array-like): Digital Elevation Model data.
    real_WD (array-like): Real water depth data.
    pred_WD (array-like): Predicted water depth data.
    """
    fig, axs = plt.subplots(1, 4, figsize=(17, 5))

    # Calculate the difference and maximum values
    diff_WD = real_WD - pred_WD
    max_WD = max(pred_WD.max(), real_WD.max())
    max_diff_WD = max(diff_WD.max(), -diff_WD.min())

    # Custom colormap that starts with white and then uses 'Blues'
    cmap = plt.cm.Blues
    cmap.set_under('white', alpha=0)

    axs[0].imshow(DEM.squeeze(), cmap='terrain', origin='lower')
    axs[1].imshow(real_WD.squeeze(), vmin=1e-6, vmax=2, cmap=cmap, origin='lower')
    axs[2].imshow(pred_WD.squeeze(), vmin=1e-6, vmax=2, cmap=cmap, origin='lower')
    axs[3].imshow(diff_WD.squeeze(), vmin=-max_diff_WD, vmax=max_diff_WD, cmap='RdBu', origin='lower')

    # Adding colorbars
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=DEM.min(), vmax=DEM.max()), cmap='terrain'),
                 fraction=0.05, shrink=0.9, ax=axs[0])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=1e-6, vmax=2), cmap=cmap),
                 fraction=0.05, shrink=0.9, ax=axs[1])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=1e-6, vmax=2), cmap=cmap),
                 fraction=0.05, shrink=0.9, ax=axs[2])
    plt.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-2, vmax=+2, vcenter=0),
                 cmap='RdBu'), fraction=0.05, shrink=0.9, ax=axs[3])

    for ax in axs:
        ax.axis('off')

    axs[0].set_title('DEM')
    axs[1].set_title('Real Water Depth (m)')
    axs[2].set_title('Predicted Water Depth (m)')
    axs[3].set_title('Difference (m)')

    plt.show()
    
def plot_water_depth_comparison_to_image(DEM, real_WD, pred_WD, image_path):
    """
    Generates a plot comparing real and predicted water depth along with DEM and difference,
    and saves the plot as an image without the colorbar for the DEM plot, with fixed legend scales for water depths and difference.

    Args:
    DEM (array-like): Digital Elevation Model data.
    real_WD (array-like): Real water depth data.
    pred_WD (array-like): Predicted water depth data.
    image_path (str): Path to save the plot image.
    """
    fig, axs = plt.subplots(1, 4, figsize=(17, 5), dpi=100)

    # Calculate the difference
    diff_WD = real_WD - pred_WD

    # Custom colormap that starts with white and then uses 'Blues'
    cmap = plt.cm.Blues
    cmap.set_under('white', alpha=0)

    axs[0].imshow(DEM.squeeze(), cmap='terrain', origin='lower')
    # Fixed maximum value for water depth plots
    fixed_max_WD = 2
    axs[1].imshow(real_WD.squeeze(), vmin=1e-6, vmax=fixed_max_WD, cmap=cmap, origin='lower')
    axs[2].imshow(pred_WD.squeeze(), vmin=1e-6, vmax=fixed_max_WD, cmap=cmap, origin='lower')
    # Fixed range for difference plot
    fixed_diff_range = 2
    axs[3].imshow(diff_WD.squeeze(), vmin=-fixed_diff_range, vmax=fixed_diff_range, cmap='RdBu', origin='lower')

    # Adding colorbars with fixed ranges
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=1e-6, vmax=fixed_max_WD), cmap=cmap),
                 fraction=0.05, shrink=0.9, ax=axs[1])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=1e-6, vmax=fixed_max_WD), cmap=cmap),
                 fraction=0.05, shrink=0.9, ax=axs[2])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-fixed_diff_range, vmax=fixed_diff_range),
                 cmap='RdBu'), fraction=0.05, shrink=0.9, ax=axs[3])

    for ax in axs:
        ax.axis('off')

    axs[0].set_title('DEM')
    axs[1].set_title('Real Water Depth (m)')
    axs[2].set_title('Predicted Water Depth (m)')
    axs[3].set_title('Difference (m)')

    plt.savefig(image_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

from IPython.display import Image, display
import imageio
import os

def create_gif_from_preds_and_show(preds, gif_path, duration=1.0):  # Increase duration for slower playback
    """
    Creates a GIF from a series of predictions, makes it play slower, loops indefinitely, and displays it in a Jupyter Notebook.

    Args:
    preds (list of tuples): Each tuple contains (DEM, real_WD, pred_WD).
    gif_path (str): Path to save the GIF.
    duration (float): Duration of each frame in the GIF, in seconds. Increase for slower playback.
    """
    image_paths = []

    # Generate images
    for n, (DEM, real_WD, pred_WD) in enumerate(preds):
        image_path = f'temp_frame_{n}.png'
        plot_water_depth_comparison_to_image(DEM, real_WD, pred_WD, image_path)
        image_paths.append(image_path)

    # Compile images into a GIF with infinite looping
    with imageio.get_writer(gif_path, mode='I', duration=duration, loop=0) as writer:  # loop=0 for infinite loop
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)
            # Remove the temporary image files after adding them to the GIF
            os.remove(image_path)

    # Display the GIF in the notebook
    display(Image(filename=gif_path, format='png'))
# .py file containing the functions used for plotting results from CNN/ConvLSTM models for DSAIE FLOOD project.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import os
import io
#import imageio.v2 as imageio 

from models.ConvLSTM_model.train_eval import *
from post_processing.metrics import confusion_mat
from post_processing.sort_dataset import *
from pre_processing.normalization import *
from post_processing.cool_animation import *

def plot_losses(train_losses, validation_losses, model, loss_f = 'MSE'):
    '''
    Function for plotting the training and validation losses of CNN and ConvLSTM models.
    y-axis is in log-scale. 
    
    Inputs: train_losses = list containing training losses throughout epochs
            validation_losses =  list containing validation losses throughout epochs
            model used for training
    loss_f : str
             key that specifies the function for computing the loss, 
             accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception
             default = 'MSE'

    Outputs: None 
    '''
    model_who = str(model.__class__.__name__)
    plt.figure() 
    plt.plot(train_losses, color='blue', ls='--', label='Training')
    plt.plot(validation_losses, color='red', ls='--', label='Validation')
    plt.yscale('log')
    plt.title(model_who + ' training and validation ' + loss_f)
    plt.xlabel('Epochs [-]')
    plt.ylabel('Normalized ' + loss_f + ' [-]')
    plt.gca().yaxis.set_label_coords(-0.01, 0.5)
    plt.legend()
    plt.show()

    return None

def plot_test_loss(dataset, model, train_val_test, device, loss_f='MSE'):
    '''
    Plot test loss for each sample of the requested dataset and model

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Normalized dataset of train_val or test (1, 2, or 3).
    model : class of model
        Model to predict on the sample.
    train_val_test: key for specifying what we are using the model for
                            'train_val' = train and validate the model
                            'test1' = test the model with dataset 1
                            'test2' = test the model with dataset 2
                            'test3' = test the model with dataset 3
                    Used also in the title of the animation.
    device : str
        Device on which to perform the computations
    loss_f : str
             key that specifies the function for computing the loss, 
             accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception
             default = 'MSE'

    Returns
    -------
    None.

    '''
    model_who = str(model.__class__.__name__)
    
    # get number of samples and initialize loss list
    num_samples = len(dataset)
    sample_array = np.arange(0, num_samples)
    loss_sample = np.zeros((num_samples))
    
    # make predictions for each sample
    for sample in range(num_samples):
        target = dataset[sample][1]
        time_steps = target.shape[0]
        preds = obtain_predictions(model, dataset[sample][0], device, time_steps)
        
        loss_sample[sample] = choose_loss(loss_f, preds, target)
    
    plt.figure()
    plt.scatter(sample_array, loss_sample)
    plt.xticks(range(num_samples))
    plt.xlabel('Sample Number')
    plt.ylabel('Normalized '+ loss_f + ' [-]')
    plt.title(model_who + ': ' + train_val_test + ' ' + loss_f + ' for each sample')
    plt.grid()
    plt.show()
    
    return None

def plot_sorted(dataset, model, train_val_test, scaler_x, scaler_y, device,
                thresholds = torch.tensor([0.1, 0]).reshape(1, -1), loss_f = 'MSE', loss_recall='loss'):
    '''
    Function for plotting the DEMs variation sorted in increasing order 
    of average loss (of Water Depth and Discharge), the relative Water Depth and 
    Discharge loss and the relative Recall

    Input: dataset = tensor, normalized dataset
           model = trained AI model
           train_val_test: key for specifying what we are using the model for
                            'train_val' = train and validate the model
                            'test1' = test the model with dataset 1
                            'test2' = test the model with dataset 2
                            'test3' = test the model with dataset 3
           scaler_x, scaler_y = scalers for inputs (x) and targets (water depth and discharge), created 
                                            with the scaler function 
           device = device on which running the simulations, accepts 'cpu' or 'cuda'
           thresholds: torch.tensor, Denormalized thresholds for each feature. Expects a tensor that 
                        has shape: (1 x num_features). 
                        default = 0.1 m
           loss_f = str, key that specifies the function for computing the loss, 
                    accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception
                    default = 'MSE'
           loss_recall : srt,
                  if best_worst is not None this specifies if samples are sorted 
                  based on average loss or recall. 
                  Expects 'None', 'loss' or 'recall'.
                  default = 'loss'
                    
    Output: None (plot)
    '''
    model_who = str(model.__class__.__name__)
    n_samples = len(dataset)
    # n_features = dataset[0][1].shape[1]
    # n_pixels = dataset[0][1].shape[-1]
    # time_steps = dataset[0][1].shape[0]

    sorted_indexes, elevation, losses, recall = get_indexes(
          dataset, model, train_val_test, scaler_x, scaler_y, 
          device, thresholds, loss_f, loss_recall)
    
    sorted_loss = losses[sorted_indexes, :]
    denorm_loss = scaler_y.inverse_transform(sorted_loss) # Loss now contains units
    
    elevation_sorted = elevation[sorted_indexes]
    elev_sorted_tensor = torch.Tensor(elevation_sorted)
    
    sorted_recall = recall[sorted_indexes]
    
    # plot 
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True) 
    plt.subplots_adjust(hspace=0.5)

    # create second y-axis for discharge scale
    ax1_2 = axes[1].twinx()
    
    sample_list = np.arange(0, n_samples)
    
    axes[0].boxplot(elev_sorted_tensor.reshape(n_samples, -1), positions = range(0, n_samples), showfliers=False)  
    axes[1].scatter(sample_list, denorm_loss[:, 0], color='blue', marker='x', label='water depth') 
    ax1_2.scatter(sample_list, denorm_loss[:, 1], color='red', marker='x', label='discharge') 
    axes[2].scatter(sample_list, sorted_recall, color='green', marker='x', label='recall') 

    labels = np.array([sorted_indexes[i] for i in sorted_indexes])
    
    # since MSE is not square rooted we need to be correct with out units when denormalizing
    if loss_f == 'MAE':
        squared = ''
    elif loss_f == 'MSE':
        squared = r'$^2$'
    
    for ax in axes:
        ax.set_xlabel('Sample ID')
        ax.set_xticks(sorted_indexes, labels, rotation=45, ha='right')
        
    fig.legend(bbox_to_anchor=(0.27, 0.39))

    axes[0].set_ylabel('Elevation [m]')
    axes[1].set_ylabel(r'Water Depth loss $[m]$' + squared)
    ax1_2.set_ylabel(r'Discharge loss $[m^{3} s^{-1} m^{-1}]$' + squared)
    axes[2].set_ylabel('Recall [-]')

    axes[0].set_title('DEM elevation and variation')
    axes[1].set_title(loss_f)
    axes[2].set_title(f'Water Depth Recall\n\
with a minimum threshold of {thresholds[0,0]:.2f} m')

    fig.suptitle(train_val_test + ': Peformance of ' + model_who + ' with respect to the variablity of the DEM', fontsize=15)
    plt.xlim(-1, n_samples+1)
    plt.show()
    return None 

def plot_metrics(dataset, model, train_val_test, scaler_y, device,
                 thresholds = torch.tensor([0.1, 0]).reshape(1, -1)):
    '''
    Function for plotting different metrics (recall, precision and F1-score)
    
    Inputs: dataset = TensorDataset, normalized dataset containing inputs and targets
            model = trained AI model
            train_val_test: key for specifying what we are using the model for
                            'train_val' = train and validate the model
                            'test1' = test the model with dataset 1
                            'test2' = test the model with dataset 2
                            'test3' = test the model with dataset 3
            scaler_y = scaler of targets (water depth and discharge), created with the scaler function  
            device = device on which running the simulations, accepts 'cpu' or 'cuda' 
            thresholds: torch.tensor, Denormalized thresholds for each feature. Expects a tensor that 
                        has shape: (1 x num_features). 
                        default = 0.1 m

    Outputs: None
    '''

    # get the model name
    model_who = str(model.__class__.__name__)

    # compute metrics
    recall, accuracy, f1 = confusion_mat(dataset, model,
                                         scaler_y, device, thresholds)
    
    # get number of samples
    num_samples = len(dataset)
    sample_array = np.arange(0, num_samples)
    
    plt.figure()
    plt.scatter(sample_array, recall, label = 'Recall')
    plt.scatter(sample_array, accuracy, label = 'Accuracy')
    plt.scatter(sample_array, f1, label = 'F1')
    plt.xticks(range(num_samples))
    plt.xlabel('Sample Number')
    plt.ylabel('Score [-]')
    plt.title(model_who + ': ' + train_val_test + ' metrics for each sample (WD only)')
    plt.legend()
    plt.grid()
    plt.show()
    return None


def create_combined_gif(actual_array, predicted_array, difference_array, dem_map, title, dataset_name, cmap='Blues', loop=0, figsize=(4, 4)):
    """
    Creates a combined GIF with DEM map and multiple subplots for actual, predicted, and difference arrays.

    Parameters:
    - actual_array (numpy.ndarray): The actual data array.
    - predicted_array (numpy.ndarray): The predicted data array.
    - difference_array (numpy.ndarray): The difference data array.
    - dem_map (numpy.ndarray): The DEM (Digital Elevation Map) array.
    - title (str): Title for the GIF and subplots.
    - dataset_name (str): Name of the dataset for saving the GIF file.
    - cmap (str): Colormap for actual and predicted arrays. Default is 'Blues'.
    - loop (int): Number of times to loop the GIF. Default is 0 (no loop).
    - figsize (tuple): Figure size for subplots. Default is (4, 4).

    Returns:
    - str: Filename of the saved GIF.
    """
    colormaps = ['terrain', 'coolwarm', 'coolwarm', 'Blues']
    # Calculate the constant min and max values for the color scale
    min_value = min(difference_array.min(), 0)  # Minimum value set to 0
    max_value = max(actual_array.max(), predicted_array.max(), difference_array.max())

    images = []
    for t in range(actual_array.shape[0]):
        fig, axs = plt.subplots(1, 4, figsize=(4 * figsize[0], figsize[1]))  # Create four subplots side by side
        im_dem = axs[0].imshow(dem_map, cmap=colormaps[0], origin='lower', vmin=min_value, vmax=max_value)  # Display DEM map
        im_actual = axs[1].imshow(actual_array[t], cmap=cmap, origin='lower', vmin=0, vmax=max_value)  # Start from 0 as white
        im_predicted = axs[2].imshow(predicted_array[t], cmap=cmap, origin='lower', vmin=0, vmax=max_value)  # Start from 0 as white  

        # changed im_difference to make sure the scale is symmetric around zero
        im_difference = axs[3].imshow(difference_array[t], cmap='seismic', origin='lower', vmin=-max_value, vmax=max_value)  # Using a specific colormap for difference

        fig.colorbar(im_dem, ax=axs[0], fraction=0.046, pad=0.04)
        fig.colorbar(im_actual, ax=axs[1], fraction=0.046, pad=0.04)
        fig.colorbar(im_predicted, ax=axs[2], fraction=0.046, pad=0.04)
        fig.colorbar(im_difference, ax=axs[3], fraction=0.046, pad=0.04)

        axs[0].set_title('DEM Map')  # Title for DEM map
        axs[1].set_title(f"Actual {title} {t+1}")
        axs[2].set_title(f"Predicted {title} {t+1}")
        axs[3].set_title(f"Difference {title} {t+1}")

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()

        # Convert the figure to an image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        plt.close()

    # Specify the folder for saving the GIF
    save_folder = 'post_processing'
    os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

    gif_filename = os.path.join(save_folder, f'combined_{title.replace(" ", "_").lower()}_{dataset_name}.gif')
    with imageio.get_writer(gif_filename, mode='I', loop=loop) as writer:
        for image in images:
            writer.append_data(image)

    return gif_filename
# .py file containing the functions used for plotting results from CNN/ConvLSTM models for DSAIE FLOOD project.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.ConvLSTM_model.train_eval import *
from post_processing.metrics import confusion_mat
from post_processing.sort_dataset import *
from pre_processing.normalization import *
from post_processing.cool_animation import *

def plot_losses(train_losses, validation_losses, model):
    '''
    Function for plotting the training and validation losses of CNN and ConvLSTM models.
    y-axis is in log-scale. 
    
    Inputs: train_losses = list containing training losses throughout epochs
            validation_losses =  list containing validation losses throughout epochs
            model used for training

    Outputs: None 
    '''
    model_who = str(model.__class__.__name__)
    plt.figure() 
    plt.plot(train_losses, color='blue', ls='--', label='Training')
    plt.plot(validation_losses, color='red', ls='--', label='Validation')
    plt.yscale('log')
    plt.title(model_who + ' training and validation losses')
    plt.xlabel('Epochs [-]')
    plt.ylabel('Normalized Loss [-]')
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
    plt.ylabel('Normalized Loss [-]')
    plt.title(model_who + ': ' + train_val_test + ' losses for each sample')
    plt.grid()
    plt.show()
    
    return None

def plot_sorted(dataset, model, train_val_test, scaler_x, scaler_y, device,
                thresholds = torch.tensor([0.1, 0]).reshape(1, -1), loss_f = 'MSE', loss_recall='None'):
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
                  default = 'None'
                    
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
    
    for ax in axes:
        ax.set_xlabel('Sample ID')
        ax.set_xticks(sorted_indexes, labels, rotation=45, ha='right')
        
    fig.legend(bbox_to_anchor=(0.27, 0.67))

    axes[0].set_ylabel('Elevation [m]')
    axes[1].set_ylabel(r'Water Depth loss [$m^2$]')
    ax1_2.set_ylabel(r'Discharge loss [$m^{3} s^{-1} m^{-1}$]')
    axes[2].set_ylabel('Recall [-]')

    axes[0].set_title('DEM elevation and variation')
    axes[1].set_title('MSE loss')
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

def plot_best_worst(dataset, model, train_val_test, scaler_x, scaler_y, device,
                    thresholds = torch.tensor([0.1, 0]).reshape(1, -1), 
                    loss_f = 'MSE', best_worst='best', loss_recall='loss', save=False):
    '''
       Function for plotting the best and worst samples. 
       The definition of best and worst is based on the value of the average loss (between Water Depth and Discharge)
       and the value of computed recall for a specific dataset.
       
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
                    
    Output: None (plot)
    '''
    # get inputs and outputs
    # 1st sample, 2nd input(0)/target(1), 3rd time step, 4th features, 5th/6th pixels
    model_who = str(model.__class__.__name__)
    n_samples = len(dataset)
    n_features = dataset[0][1].shape[1]
    n_pixels = dataset[0][1].shape[-1]
    time_steps = dataset[0][1].shape[0]
    
    # initialize inputs and outputs
    inputs = []
    targets = []
    
    for i in range(n_samples):
        inputs.append(dataset[i][0])
        targets.append(dataset[i][1])

    # initialize denormalization of dataset
    elevation = np.zeros((n_samples, n_pixels, n_pixels))

    # initialize losses
    losses = torch.zeros((n_samples, n_features))

    for i in range(n_samples):
        # denormalize elevation
        elevation[i], _, _ = denormalize_dataset(inputs[i], targets[i], train_val_test,
                                                            scaler_x, scaler_y)
        # make predictions
        preds = obtain_predictions(model, inputs[i], device, time_steps)

    # compute MSE losses
    for feature in range(n_features):
            for i in range(n_samples):
                 losses[i, feature] = choose_loss(loss_f, preds[:][feature], targets[i][:][feature])
    
    # average over columns = features
    avg_loss = torch.mean(losses, dim=1)

    recall, _, _ = confusion_mat(dataset, model, scaler_y, device, thresholds)

    # sorting dataset
    if loss_recall == 'loss':
         _, sorted_indexes = torch.sort(avg_loss)
    elif loss_recall == 'recall':
         _, sorted_indexes = torch.sort(recall)
    
    sorted_loss = losses[sorted_indexes, :]
    denorm_loss = scaler_y.inverse_transform(sorted_loss) # Loss now contains units
    
    elevation_sorted = elevation[sorted_indexes]
    elev_sorted_tensor = torch.Tensor(elevation_sorted)
    
    sorted_recall = recall[sorted_indexes]
    
    # get best or worst model
    if best_worst == 'best':
         idx = -1
    elif best_worst == 'worst':
         idx = 1
    
    if loss_recall == 'loss':
         value = sorted_loss[idx]
    elif loss_recall == 'recall':
         value = sorted_recall[idx]

    plot_animation(idx, dataset, model, train_val_test, scaler_x,
                   scaler_y, device, save, thresholds, loss_f)
    return None
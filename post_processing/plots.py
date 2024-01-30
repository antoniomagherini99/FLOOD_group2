# .py file containing the functions used for plotting results from CNN/ConvLSTM models for DSAIE FLOOD project.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.ConvLSTM_model.train_eval import obtain_predictions
from post_processing.metrics import confusion_mat
from pre_processing.normalization import *

def plot_losses(train_losses, validation_losses, model):
    '''
    Function for plotting the training and validation losses of CNN and ConvLSTM models.
    y-axis is in log-scale. 
    
    Inputs: train_losses = list containing training losses throughout epochs
            validation_losses =  list containing validation losses throughout epochs
            model = 'CNN' or 'ConvLSTM', key that specifies model used 

    Outputs: None 
    '''

    plt.figure() 
    plt.plot(train_losses, color='blue', ls='--', label='Training')
    plt.plot(validation_losses, color='red', ls='--', label='Validation')
    plt.yscale('log')
    plt.title(model + 'training and validation losses')
    plt.xlabel('Epochs [-]')
    plt.ylabel('Nromalized Loss [-]')
    plt.gca().yaxis.set_label_coords(-0.01, 0.5)
    plt.legend()
    plt.show()

    return None

def plot_test_loss(dataset, model, train_val, device):
    '''
    Plot test loss for each sample of the requested dataset and model

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Normalized dataset of train_val or test (1, 2, or 3).
    model : class of model
        Model to predict on the sample.
    train_val : str
        key for specifying what we are using the model for
            'train_val' = train and validate the model
            'test1' = test the model with dataset 1
            'test2' = test the model with dataset 2
            'test3' = test the model with dataset 3
        Used in the title of the animation.
    device : str
        Device on which to perform the computations

    Returns
    -------
    None.

    '''
    model_who = str(model.__class__.__name__)
    
    num_samples = len(dataset)
    sample_array = np.arange(0, num_samples)
    loss_sample = np.zeros((num_samples))
    
    for sample in range(num_samples):
        target = dataset[sample][1]
        time_steps = target.shape[0]
        preds = obtain_predictions(model, dataset[sample][0], device, time_steps)
        
        loss_sample[sample] = nn.MSELoss()(preds, target)
    
    plt.figure()
    plt.scatter(sample_array, loss_sample)
    plt.xticks(range(num_samples))
    plt.xlabel('Sample Number')
    plt.ylabel('Normalized Loss [-]')
    plt.title(model_who + ': ' + train_val + ' losses for each sample')
    plt.grid()
    plt.show()
    
    return None

def plot_sorted(dataset, model, train_val, scaler_x, scaler_y, device,
                thresholds = torch.tensor([0.1, 0]).reshape(1, -1)):
    '''
    Function for plotting the DEMs variation sorted in increasing order 
    of average loss (of Water Depth and Discharge)

    Input: dataset = tensor, normalized dataset
           train_val_test : str, Identifier of dictionary. Expects: 'train_val', 'test1', 'test2', 'test3'.
           scaler_x, scaler_y = scalers for inputs (x) and targets (water depth and discharge), created 
                                            with the scaler function 
    Output: None (plot)
    '''

    # get inputs and outputs
    # 1st sample, 2nd input(0)/target(1), 3rd time step, 4th features, 5th/6th pixels
    model_who = str(model.__class__.__name__)
    n_samples = len(dataset)
    n_features = dataset[0][1].shape[1]
    n_pixels = dataset[0][1].shape[-1]
    time_steps = dataset[0][1].shape[0]

    # print(f'Samples: {n_samples}')
    # print(f'Features: {n_features}')
    # print(f'Pixels: {n_pixels}')
    # print(f'Time steps: {time_steps}\n')
    
    # initialize inputs and outputs
    inputs = []
    targets = []
    
    for i in range(n_samples):
        inputs.append(dataset[i][0])
        targets.append(dataset[i][1])

    # print(f'Inputs shape: {np.shape(inputs[0])}, targets shape: {np.shape(targets[0])}')
    # print(f'Input type: {type(inputs)}\n')

    # initialize denormalization of dataset
    elevation = np.zeros((n_samples, n_pixels, n_pixels))

    # initialize losses
    losses = torch.zeros((n_samples, n_features))

    for i in range(n_samples):
        # denormalize elevation
        elevation[i], _, _ = denormalize_dataset(inputs[i], targets[i], train_val,
                                                            scaler_x, scaler_y)
        # make predictions
        preds = obtain_predictions(model, inputs[i], device, time_steps)
    
    # print(f'Elevation {elevation}, shape: {elevation.shape}, type: {type(elevation)}')
    # print(f'water_depth shape: {water_depth.shape}, type: {type(water_depth)}')
    # print(f'discharge shape: {discharge.shape}, type: {type(discharge)}')
    # print(f'predictions shape: {preds.shape}, type: {type(preds)}')
    # print(f'predictions wd shape: {preds[:,0].shape}, type: {type(preds[:,0])}')
    # print(f'predictions q shape: {preds[:,0].shape}, type: {type(preds[:,0])}\n')

    # compute MSE losses
    for feature in range(n_features):
            for i in range(n_samples):
                 losses[i, feature] = nn.MSELoss()(preds[:][feature], targets[i][:][feature])
    # print(f'losses shape: {losses.shape}, type: {type(losses)}\n')
    
    # average over columns = features
    avg_loss = torch.mean(losses, dim=1)
    # print(f'Avg loss shape: {avg_loss.shape}, type:{type(avg_loss)}\n')

    # compute recall - improvement: add minimium threshold for recall (wd > 10 cm), need to denormalize targets and predictions
    # ask scaler what 10 is and plot that scaler_wd.transform(0.10) - check
    
    recall, _, _ = confusion_mat(dataset, model, scaler_y, device, thresholds)
    # print(f'recall: {recall}, shape: {recall.shape}\n')

    # sorting dataset
    _, sorted_indexes = torch.sort(avg_loss)
    
    sorted_loss = losses[sorted_indexes, :]
    denorm_loss = scaler_y.inverse_transform(sorted_loss) # Loss now contains units
    
    # sorted_loss = []
    # for i in range(n_samples):
    #      for j in range(n_features):
    #           sorted_loss[i, j] = sorted_indexes[i]
    # print(f'sorted_indexes: {sorted_indexes},\n\
# sorted_loss wd: {sorted_l_wd},\n\sorted_loss q: {sorted_l_q}')
    # print(f'sorted_indexes shape: {sorted_indexes.shape}, sorted_loss wd shape: {sorted_l_wd.shape}, sorted_loss q shape: {sorted_l_q.shape}\n')
    elevation_sorted = elevation[sorted_indexes]
    elev_sorted_tensor = torch.Tensor(elevation_sorted)
    # print(len(elevation_sorted[:][:][:]))
    # print(elevation_sorted[:][:][:])
    # print(f'Elevation sorted shape: {elevation_sorted.shape}, type: {type(elevation_sorted)}')
    # print(f'Check if they are the same {elevation==elevation_sorted}')
    
    # compute mean over x- and y-direction 
    #mean_elev = torch.mean(elev_sorted_tensor, dim=(1,2))
    # compute variation
    #var_elev = elev_sorted_tensor - mean_elev[:, np.newaxis, np.newaxis]
    
    # not needed actually
    # wd_sorted = [water_depth[i] for i in sorted_indexes]
    # q_sorted  = [discharge[i] for i in sorted_indexes]   
    # print(f'wd_sorted shape: {wd_sorted.shape}, type: {type(wd_sorted)}')
    # print(f'Check if they are the same {water_depth==wd_sorted}\n')
    # print(f'wd_sorted shape: {q_sorted.shape}, type: {type(q_sorted)}')
    # print(f'Check if they are the same {discharge==q_sorted}\ng')
    
    sorted_recall = recall[sorted_indexes]
    # print(f'sorted_recall shape: {sorted_recall.shape}, type: {type(sorted_recall)}')
    # print(f'Check if they are the same {recall==sorted_recall}\n')
    
    # plot 
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True) 
    plt.subplots_adjust(hspace=0.5)

    # create second y-axis for discharge scale
    ax1_2 = axes[1].twinx()
    
    sample_list = np.arange(0, n_samples)
    
    axes[0].boxplot(elev_sorted_tensor.reshape(n_samples, -1), positions = range(0, n_samples), showfliers=False)  #(var_elev.reshape(128, -1), showfliers=False) 
    axes[1].scatter(sample_list, denorm_loss[:, 0], color='blue', marker='x', label='water depth') 
    ax1_2.scatter(sample_list, denorm_loss[:, 1], color='red', marker='x', label='discharge') 
    axes[2].scatter(sample_list, sorted_recall, color='green', marker='x', label='recall') 

    labels = np.array([sorted_indexes[i] for i in sorted_indexes])
    
    for ax in axes:
        ax.set_xlabel('Sample ID')
        ax.set_xticks(sorted_indexes, labels, rotation=45, ha='right')
        
    fig.legend(bbox_to_anchor=(0.27, 0.67))

     
    axes[0].set_ylabel('Elevation [m]')
    axes[1].set_ylabel('Water Depth loss [m]')
    ax1_2.set_ylabel(r'Discharge loss [$m^{3} s^{-1} m^{-1}$]')
    axes[2].set_ylabel('Recall [-]')

    axes[0].set_title('DEM elevation and variation')
    axes[1].set_title('MSE loss')
    axes[2].set_title('Water Depth Recall')

    fig.suptitle(train_val + ': Peformance of ' + model_who + ' with respect to the variablity of the DEM', fontsize=15)
    plt.xlim(-1, n_samples+1)
    plt.show()
    return None

def plot_metrics(dataset, model, train_val, scaler_y, device,
                 thresholds = torch.tensor([0.1, 0]).reshape(1, -1)):
    model_who = str(model.__class__.__name__)
    recall, accuracy, f1 = confusion_mat(dataset, model,
                                         scaler_y, device, thresholds)
    
    num_samples = len(dataset)
    sample_array = np.arange(0, num_samples)
    
    plt.figure()
    plt.scatter(sample_array, recall, label = 'Recall')
    plt.scatter(sample_array, accuracy, label = 'Accuracy')
    plt.scatter(sample_array, f1, label = 'F1')
    plt.xticks(range(num_samples))
    plt.xlabel('Sample Number')
    plt.ylabel('Score [-]')
    plt.title(model_who + ': ' + train_val + ' metrics for each sample (WD only)')
    plt.legend()
    plt.grid()
    plt.show()
    return None
        

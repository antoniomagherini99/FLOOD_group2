import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import torch
import numpy as np

from pre_processing.normalization import denormalize_dataset
from post_processing.sort_dataset import * 
from post_processing.metrics import confusion_mat
from models.ConvLSTM_model.train_eval import *

def definitions(index):
    '''
    Function that contains dictionaries to keep the naming and units constant
    in plots

    Parameters
    ----------
    index : int
        0 represents water depth
        1 represents discharge

    Returns
    -------
    name : str
        Name of variable
    unit : str
        Units of variable

    '''
    feature_dic = {
        0: 'Water Depth',
        1: 'Discharge per Meter Width'
    }

    feature_dic_units = {
        0: r'$m$',
        1: r'$m^3 s^{-1} m^{-1}$'
    }
    if index != 0 and index !=1:
        raise Exception('Variable "index" must be 0 or 1')
    else:
        name = feature_dic[index]
        unit = feature_dic_units[index]
    return name, unit

def find_axes(axis):
    '''
    Define which axes will be used to place the colorbar

    Parameters
    ----------
    axis : instance of an axes class of matplotlib
        subplot that needs to have axes locatable to place colorbar

    Returns
    -------
    cax : instance of an axes class of matplotlib
        axes into which the colorbar will be drawn

    '''
    div = make_axes_locatable(axis)
    cax = div.append_axes('right', '5%', '5%')
    return cax

def animated_plot(figure, animated_tensor, axis,
                  variable, fontsize, diff = False, prediction = False):
    '''
    Function used to generalise animated plots

    Parameters
    ----------
    figure : instance of a figure class of matplotlib
        Figure used in the subplots
    animated_tensor : torch.tensor
        Tensor that needs to be animated. Shape is (time_steps x num_features x
        pixel x pixel) 
    axis : instance of an axes class of matplotlib
        Subplot that will contain the animated tensor
    variable : str
        Name of variable that will be animated
        Two options available: "water_depth" and "discharge"
    fontsize: int
        Sets the fontsize of the title
    diff : bool, optional
        Identifier that defines whether the subplot is used for a difference.
        The default is False.
    prediction : bool, optional
        Identifier that defines whether the subplot is used for a prediction
        or a target.
        The default is False.

    Returns
    -------
    image : instance of an image class of matplotlib
        Image that will be updated in the "animation" function for each time
        step.

    '''
    # Set color maps, units and titles
    if variable == 'water_depth':
        cmap = 'Blues'
        title, label = definitions(0)
    elif variable == 'discharge':
        cmap = 'Greens'
        title, label = definitions(1)
    else:
        raise Exception ('Check variable input. "water_depth" and ' +
                         '"discharge" only valid inputs')  
    
    if diff == False:
        if prediction == False:
            title = 'Targets: ' + title
        elif prediction == True:
            title = 'Predictions: ' + title
        else:
            raise TypeError ('"Prediction" needs to be a boolean')
    elif diff == True:
        cmap = 'seismic' 
        title = 'Differences: ' + title
    else:
        raise TypeError ('"diff" needs to be a boolean')
        
    # Start plotting
    cax = find_axes(axis)
    image = axis.imshow(animated_tensor[0], cmap=cmap, origin='lower')
    cb = figure.colorbar(image, cax=cax)
    cb.set_label(label)
    axis.set_title(title, fontsize = fontsize, fontweight='bold')
    # set color bar limits
    min_val = animated_tensor.min()
    max_val = animated_tensor.max()
    if diff == True:
        absolute_max = np.max(np.array([abs(min_val), max_val]))
        min_val = -1 * absolute_max
        max_val = absolute_max
    elif diff == False:
        None
    else:
        raise TypeError ('"diff" needs to be a boolean')
        
    image.set_clim(min_val, max_val)
    return image

def plot_dem(figure, DEM, boundary_condition, axis, fontsize):
    '''
    Plot the elevation with the location of the boundary condition as an x.

    Parameters
    ----------
    figure : instance of a figure class of matplotlib
        Figure used in the subplots
    DEM : torch.Tensor
        pixel x pixel tensor containing the elevation of the sample
    boundary_condition : torch.Tensor
        pixel x pixel tensor containing the location of the boundary condition
        of the sample
    axis : instance of an axes class of matplotlib
        Subplot that will contain the animated tensor
    fontsize: int
        Sets the fontsize of the title

    Returns
    -------
    None.

    '''
    
    cax = find_axes(axis)
    image = axis.imshow(DEM, cmap='terrain', origin='lower')
    cb = figure.colorbar(image, cax=cax)
    cb.set_label(r'$m$')
    axis.set_title('Elevation, X = Breach loc.', fontsize = fontsize, fontweight='bold')
    non_zero_indices = torch.nonzero(boundary_condition)
    non_zero_row, non_zero_col = non_zero_indices[0][0].item(), non_zero_indices[0][1].item()
    axis.scatter(non_zero_col, non_zero_row, color='k', marker='x', s=100,
                clip_on = False, clip_box = plt.gca().transData)
    return None

def plot_loss_per_hour(targets, predictions, axis, loss_f, fontsize):
    '''
    Plot the loss per hour based on the choice of loss function, dataset, 
    sample and model

    Parameters
    ----------
    targets : torch.Tensor
        Targets of the sample of the datset. 
        Shape is (time_steps x num_features x pixel x pixel)
    predictions : torch.Tensor
        Predictions of the model for a sample.
        Shape is (time_steps x num_features x pixel x pixel)
    axis : instance of an axes class of matplotlib
        Subplot that will contain the animated tensor
    loss_f : str
             key that specifies the function for computing the loss, 
             accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception.
    fontsize: int
        Sets the fontsize of the title

    Returns
    -------
    None.

    '''
    features = targets.shape[1]
    time_steps = targets.shape[0]
    
    losses = np.zeros((features, time_steps)) # initialize empty array
    time_step_array = np.arange(1, time_steps + 1)
    
    # compute losses
    for step in range(time_steps):
        for feature in range(features):
            losses[feature, step] = choose_loss(
                loss_f, predictions[step][feature], targets[step][feature])
    
    wd_label, _ = definitions(0)
    q_label, _ = definitions(1)
    labels = [wd_label, q_label[:9]] # 9 hardcoded to reduce clutter in graph
    
    # Start Plotting
    axis.set_box_aspect(1)
    for i in range(features):
        axis.plot(time_step_array, losses[i], label = labels[i])
    axis.set_title(loss_f + ' per Hour', fontsize = fontsize, fontweight='bold')
    axis.set_xlabel('Time steps, hours since breach')
    axis.set_ylabel('Normalized ' + loss_f + ' [-]')
    axis.legend()
    return None

def plot_metric_per_hour(metric, targets, thresholds, axis, fontsize):
    '''
    Plot a metric per hour for a given sample of dataset based on a model

    Parameters
    ----------
    metric : torch.Tensor
        Options include recall, f1, and accuracy which can be calculated
        from confusion mat function. Shape is (num_features x pixel x pixel)
    targets : torch.Tensor
        Targets of the dataset. Shape is (time_steps x num_features x pixel x 
        pixel)
    thresholds: torch.Tensor
        Denormalized thresholds for each feature. Expects a tensor that
        has shape: (1 x num_features).
    axis : instance of an axes class of matplotlib
        Subplot that will contain the animated tensor
    fontsize: int
        Sets the fontsize of the title

    Returns
    -------
    None.

    '''
    time_steps = targets.shape[0]
    num_features = targets.shape[1]
    time_step_array = np.arange(1, time_steps + 1)

    wd_label, _ = definitions(0)
    q_label, _ = definitions(1)
    labels = [wd_label, q_label[:9]] # 9 hardcoded to reduce clutter in graph
    
    # Start Plotting
    axis.set_box_aspect(1)
    for i in range(num_features):
        axis.plot(time_step_array, metric[i], label = labels[i])
    axis.set_title(f'Recall/Hour with WD > {thresholds[0, 0]:.2f} m', fontsize = fontsize, fontweight='bold')
    axis.set_xlabel('Time steps, hours since breach')
    axis.set_ylabel('Recall [-]')
    axis.legend()
    return None

def plot_animation(sample, dataset, model, train_val_test, scaler_x,
                   scaler_y, device='cuda', save=False,
                   thresholds = torch.tensor([0.1, 0]).reshape(1, -1), 
                   loss_f = 'MSE', best_worst='None', loss_recall='None'):
    '''
    Plot animation to visualize the evolution of certain variables over time.
    Assumes that the model can output water depth and discharge.
    Also assumes the data is normalized.

    Parameters
    ----------
    sample : int
        Choose a sample to animate.
    dataset : torch.utils.data.Dataset
        Normalized dataset of train_val or test (1, 2, or 3).
    model : class of model
        Model to predict on the sample.
    train_val_test : str
        key for specifying what we are using the model for
            'train_val' = train and validate the model
            'test1' = test the model with dataset 1
            'test2' = test the model with dataset 2
            'test3' = test the model with dataset 3
        Used in the title of the animation.
    scaler_x : instance of normalizer
        Used on inputs.
    scaler_y : instance of normalizer
        Used on the targets/outputs
    device : str
        Device on which to perform the computations; 'cuda' is the default.
    save : bool
        Default is False. If True, will save the animation in the
        post_processing folder with the title in the format:
        'train_val + model_who + sample'.
    thresholds: torch.Tensor
        Denormalized thresholds for each feature. Expects a tensor that
        has shape: (1 x num_features).
        The default is torch.tensor([0.1, 0]).reshape(1, -1).
    loss_f : str
             key that specifies the function for computing the loss, 
             accepts 'MSE' and 'MAE'. If other arguments are set it raises an Exception
             default = 'MSE'
    best_worst : str,
                 key that specifies which sample to plot. Expects 'None', 'best', 'worst'.
                 If set to 'best' it plots the sample with the best performances, 
                 if set to 'worst' the one with worst performances 
                 based on the parameter specified with the 'loss_recall' key.
                 default = 'None'
    loss_recall : srt,
                  if best_worst is not None this specifies if samples are sorted 
                  based on average loss or recall. 
                  Expects 'None', 'loss' or 'recall'.
                  default = 'None'

    Returns
    -------
    None.
    '''

    # set fontsize for plots
    fontsize = 10

    if best_worst != 'None' and loss_recall != 'None':
         sorted_indexes, _, _, _ = get_indexes(dataset, model, train_val_test, 
                                            scaler_x, scaler_y, device, thresholds, 
                                            loss_f, loss_recall=loss_recall)
         # get best or worst model
         if best_worst == 'best':
             sample = sorted_indexes[-1]
         elif best_worst == 'worst':
             sample = sorted_indexes[1]
    elif (best_worst != 'None' and loss_recall == 'None') or (best_worst == 'None' and loss_recall != 'None'):
        raise Exception(f'Wrong keys specified!\n\
You set "best_worst" = {best_worst} and "loss_recall" = {loss_recall}, while both need to be either "None" or specified with their relative arguments.\n\
Set both keys to "None" if you do not want to get the best or worst sample or specify the wrong key with the accepted arguments.')
    
    # Extracting information from the dataset
    input = dataset[sample][0]
    target = dataset[sample][1]
    boundary_condition = input[0][3]
    
    time_steps = target.shape[0]

    # Denormalizing the data for plotting
    elevation, water_depth, discharge = denormalize_dataset(
        input, target, train_val_test, scaler_x, scaler_y)
    
    model_who = str(model.__class__.__name__)
    preds = obtain_predictions(model, input, device, time_steps)
        
    _, wd_pred, q_pred = denormalize_dataset(input, preds, train_val_test, scaler_x, scaler_y)

    # Creating subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
        3, 3, figsize=(10, 10))
    
    fig.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

    # Subplot 1
    plot_dem(fig, elevation, boundary_condition, ax1, fontsize)

    # Subplot 4
    plot_loss_per_hour(target, preds, ax4, loss_f, fontsize)
    
    # Subplot 7
    recall, _, _ = confusion_mat(dataset, model, scaler_y,
                                 device, thresholds, True, sample)

    plot_metric_per_hour(recall, target, thresholds, ax7, fontsize)

    # Subplot 2
    im2 = animated_plot(fig, water_depth, ax2, 'water_depth', fontsize)

    # Subplot 3
    im3 = animated_plot(fig, discharge, ax3, 'discharge', fontsize)

    # Subplot 5
    im5 = animated_plot(fig, wd_pred, ax5, 'water_depth', fontsize, False, True)

    # Subplot 6
    im6 = animated_plot(fig, q_pred, ax6, 'discharge', fontsize, False, True)

    # Subplot 8
    diff_wd = wd_pred - water_depth
    im8 = animated_plot(fig, diff_wd, ax8, 'water_depth', fontsize, True)

    # Subplot 9
    diff_q = q_pred - discharge
    im9 = animated_plot(fig, diff_q, ax9, 'discharge', fontsize, True)

    title_con = train_val_test + f' for sample {sample} using model: ' + model_who
    over_title = fig.suptitle('Hour 1: ' + title_con, fontsize=16) # try and update this to show the hour

    def animate(step):
        over_title.set_text(f'Hour {step + 1}: ' + title_con)
        # Subplot 2
        im2.set_data(water_depth[step])

        # 3
        im3.set_data(discharge[step])

        # 5
        im5.set_data(wd_pred[step])

        # 6
        im6.set_data(q_pred[step])

        # 8
        im8.set_data(diff_wd[step])

        # 9
        im9.set_data(diff_q[step])

    # Set up the animation
    ani = animation.FuncAnimation(fig, animate, frames = time_steps)
    plt.show()

    if save == True:
        fps = int(time_steps / 10) # Change the fps based on the amount of time steps
        ani.save('post_processing/' + train_val_test + '_' + model_who + '_' +
                 str(sample) + '.gif', writer='Pillow', fps = fps)
    elif save == False:
        None
    else:
        raise TypeError('Variable "save" must be a boolean')

    return None

def create_combined_gif(actual_array, predicted_array, difference_array,
                        dem_map, title, dataset_name, save = False,
                        figsize=(4, 4)):
    """
    Creates a combined GIF with DEM map and multiple subplots for
    actual, predicted, and difference arrays.

    Parameters:
    - actual_array (numpy.ndarray): The actual data array.
    - predicted_array (numpy.ndarray): The predicted data array.
    - difference_array (numpy.ndarray): The difference data array.
    - dem_map (numpy.ndarray): The DEM (Digital Elevation Map) array.
    - title (str): Title for the GIF and subplots.
    - dataset_name (str): Name of the dataset for saving the GIF file.
    - save (bool): save the animation if True
    - figsize (tuple): Figure size for subplots. Default is (4, 4).

    Returns:
    - str: Filename of the saved GIF.
    """
    time_steps = actual_array.shape[0]
    fontsize = 10
    # Creating subplots
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=figsize)
    
    fig.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots
    
    # Subplot 1
    #plot_dem(fig, dem_map, boundary_condition, ax1, fontsize), require BC
    
    # Subplot 2
    im2 = animated_plot(fig, actual_array, ax2, 'water_depth', fontsize)

    # Subplot 3
    im3 = animated_plot(fig, predicted_array, ax3, 'water_depth', fontsize, False, True)

    # Subplot 4
    im4 = animated_plot(fig, difference_array, ax4, 'water_depth', fontsize, True)

    title_con = dataset_name + ' for sample {sample} using model: ' + 'CNN'
    over_title = fig.suptitle('Hour 1: ' + title_con, fontsize=16) # try and update this to show the hour
    def animate(step):
        over_title.set_text(f'Hour {step + 1}: ' + title_con)
        # Subplot 2
        im2.set_data(actual_array[step])

        # 3
        im3.set_data(predicted_array[step])

        # 4
        im4.set_data(difference_array[step])

    plt.tight_layout()    
    ani = animation.FuncAnimation(fig, animate, frames = time_steps)
    plt.show()

    if save == True:
        fps = int(time_steps / 10) # Change the fps based on the amount of time steps
        ani.save('post_processing/' + dataset_name + '_CNN_' +
                 '.gif', writer='Pillow', fps = fps)

    return None

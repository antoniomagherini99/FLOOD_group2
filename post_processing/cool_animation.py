import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import torch
import torch.nn as nn
import numpy as np

from pre_processing.normalization import denormalize_dataset

def definitions(index):
    '''
    Function that contains dictionaries to keep the naming and units constant
    in plots

    Parameters
    ----------
    index : int
        0 represents water depth
        1 represents discharge
        
    Raises
    ------
    Exception
        'Variable "index" must be 0 or 1'

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
                  variable, diff = False, prediction = False):
    '''
    Function used to generalise animated plots

    Parameters
    ----------
    figure : instance of a figure class of matplotlib
        Figure used in the subplots
    animated_tensor : torch.tensor
        Tensor that needs to be animated
    axis : instance of an axes class of matplotlib
        Subplot that will contain the animated tensor
    variable : str
        Name of variable that will be animated
        Two options available: "water_depth" and "discharge"
    diff : bool, optional
        Identifier that defines whether the subplot is used for a difference.
        The default is False.
    prediction : bool, optional
        Identifier that defines whether the subplot is used for a prediction
        or a target.
        The default is False.

    Raises
    ------
    Exception
        Check "variable" input
    TypeError
        Check type of "prediction" and "diff" inputs.

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
    axis.set_title(title)
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


def plot_animation(sample, dataset, model, train_val, scaler_x,
                   scaler_wd, scaler_q, device='cuda', save=False):
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
    train_val : str
        key for specifying what we are using the model for
            'train_val' = train and validate the model
            'test1' = test the model with dataset 1
            'test2' = test the model with dataset 2
            'test3' = test the model with dataset 3
        Used in the title of the animation.
    scaler_x : instance of normalizer
        Used on inputs.
    scaler_wd : instance of normalizer
        Used on the water depth targets.
    scaler_q : instance of normalizer
        Used on the discharge target.
    device : str
        Device on which to perform the computations; 'cuda' is the default.
    save : bool
        Default is False. If True, will save the animation in the
        post_processing folder with the title in the format:
        'train_val + model_who + sample'.
    
    Raises
    ------
    Exception
        Problem with the implementation of the model, check if statements after
        the variable "model_who" is defined
    TypeError
        Check the type of save
        
    Returns
    -------
    None.
    '''

    # Extracting information from the dataset
    input = dataset[sample][0]
    target = dataset[sample][1]
    boundary_condition = input[0, 3]

    # Denormalizing the data for plotting
    elevation, water_depth, discharge = denormalize_dataset(
        input, target, train_val, scaler_x, scaler_wd, scaler_q, sample)
    
    # need to try and find a more generic way to do this
    model_who = str(model.__class__.__name__)
    if model_who == 'ConvLSTM':
        sample_list, _ = model(dataset[sample][0].unsqueeze(0).to(device))  # create a batch of 1?
        preds = torch.cat(sample_list, dim=1).detach().cpu()[0]  # remove batch
    elif model_who == 'UNet':
        preds = model(dataset[sample][0]).to(device).detach().cpu()
    else:
        raise Exception('Need to check if statements to see if model is ' +
                        'implemented correctly')
        
    _, wd_pred, q_pred = denormalize_dataset(input, preds, train_val, scaler_x, scaler_wd, scaler_q, sample)


    # Creating subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
        3, 3, figsize=(10, 10))
    
    fig.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

    # Subplot 1
    cax1 = find_axes(ax1)
    im1 = ax1.imshow(elevation, cmap='terrain', origin='lower')
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label(r'$m$')
    ax1.set_title('Elevation')

    # Subplot 4
    non_zero_indices = torch.nonzero(boundary_condition)
    non_zero_row, non_zero_col = non_zero_indices[0][0].item(), non_zero_indices[0][1].item()
    
    ax4.imshow(boundary_condition, cmap='binary', origin='lower')
    ax4.scatter(non_zero_col, non_zero_row, color='k', marker='x', s=100,
                clip_on = False, clip_box = plt.gca().transData)
    ax4.set_title('Breach Location')
    
    # Subplot 7
    features = target.shape[1]
    time_steps = target.shape[0]
    
    losses = np.zeros((features, time_steps)) # initialize empty array
    time_step_array = np.arange(1, time_steps + 1)
    
    # Compute losses uses MSELoss
    for step in range(time_steps):
        for feature in range(features):
            losses[feature, step] = nn.MSELoss()(preds[step][feature], target[step][feature])
            
    wd_label, _ = definitions(0)
    q_label, _ = definitions(1)
    
    # Start Plotting
    ax7.set_box_aspect(1)
    ax7.plot(time_step_array, losses[0], label = wd_label)
    ax7.plot(time_step_array, losses[1], label = q_label[:9]) # 9 hardcoded to reduce clutter in graph
    ax7.set_title('Losses per Hour')
    ax7.set_xlabel('Time Steps, hours')
    ax7.set_ylabel('Normalized Losses [-]')
    ax7.legend()

    # Subplot 2
    im2 = animated_plot(fig, water_depth, ax2, 'water_depth')

    # Subplot 3
    im3 = animated_plot(fig, discharge, ax3, 'discharge')

    # Subplot 5
    im5 = animated_plot(fig, wd_pred, ax5, 'water_depth', False, True)

    # Subplot 6
    im6 = animated_plot(fig, q_pred, ax6, 'discharge', False, True)

    # Subplot 8
    diff_wd = wd_pred - water_depth
    im8 = animated_plot(fig, diff_wd, ax8, 'water_depth', True)

    # Subplot 9
    diff_q = q_pred - discharge
    im9 = animated_plot(fig, diff_q, ax9, 'discharge', True)

    title_con = train_val + f' for sample {sample} using model: ' + model_who
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
        ani.save('post_processing/' + train_val + '_' + model_who + '_' +
                 str(sample) + '.gif', writer='Pillow', fps=5)
    elif save == False:
        None
    else:
        raise TypeError('Variable "save" must be a boolean')

    return None

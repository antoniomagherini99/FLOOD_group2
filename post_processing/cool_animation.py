import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import torch

from pre_processing.normalization import denormalize_dataset

def plot_animation(sample, dataset, model, title_anim, model_who, scaler_x,
                   scaler_wd, scaler_q, device='cuda', save=False):
    '''
    Plot animation to visualize the evolution of certain variables over time.

    Parameters
    ----------
    sample : int
        Choose a sample to animate.
    dataset : torch.utils.data.Dataset
        Normalized dataset of train_val or test (1, 2, or 3).
    model : class of model
        Model to predict on the sample.
    title_anim : str
        Title of the animation.
    model_who : str
        Identifies which model is used to get a prediction.
        Currently only contains "conv_lstm"; change if other models are used.
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
        'title_anim + model_who + sample'.

    Returns
    -------
    None.
    '''

    # Extracting information from the dataset
    input = dataset[sample][0]
    output = dataset[sample][1]
    boundary_condition = input[0, 3]

    # Denormalizing the data for plotting
    elevation, water_depth, discharge = denormalize_dataset(input, output, title_anim, scaler_x, scaler_wd, scaler_q, sample)

    if model_who == 'conv_lstm':
        sample_list, _ = model(dataset[sample][0].unsqueeze(0).to(device))  # create a batch of 1?
        preds = torch.cat(sample_list, dim=1).detach().cpu()[0]  # remove batch
        _, wd_pred, q_pred = denormalize_dataset(input, preds, title_anim, scaler_x, scaler_wd, scaler_q, sample)

    feature1 = 0
    feature2 = 1

    feature_dic = {
        0: 'Water Depth',
        1: 'Discharge per meter width'
    }

    feature_dic_units = {
        0: r'$m$',
        1: r'$m^2 s^{-1}$'
    }

    # Creating subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

    # Subplot 1
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', '5%', '5%')

    static_tensor = elevation
    im1 = ax1.imshow(static_tensor, cmap='terrain', origin='lower')
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label(r'$m$')
    ax1.set_title('Elevation')

    # Subplot 4
    static_tensor2 = boundary_condition
    ax4.imshow(static_tensor2, cmap='binary', origin='lower')
    ax4.set_title('Breach Location')

    # Subplot 2
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes('right', '5%', '5%')

    animated_tensor1 = water_depth
    im2 = ax2.imshow(animated_tensor1[0], cmap='Blues', origin='lower')
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label(f'{feature_dic_units[feature1]}')
    tx2 = ax2.set_title(f'Hour 0: {feature_dic[feature1]}')
    min_val1 = animated_tensor1.min()
    max_val1 = animated_tensor1.max()
    cb2.set_clim(min_val1, max_val1)

    # Subplot 3
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes('right', '5%', '5%')

    animated_tensor2 = discharge
    im3 = ax3.imshow(animated_tensor2[0], cmap='Greens', origin='lower')
    cb3 = fig.colorbar(im3, cax=cax3)
    cb3.set_label(f'{feature_dic_units[feature2]}')
    tx3 = ax3.set_title(f'Hour 0: {feature_dic[feature2]}')
    min_val2 = animated_tensor2.min()
    max_val2 = animated_tensor2.max()
    cb3.set_clim(min_val2, max_val2)

    # Subplot 5
    div5 = make_axes_locatable(ax5)
    cax5 = div5.append_axes('right', '5%', '5%')

    animated_tensor3 = wd_pred
    im5 = ax5.imshow(animated_tensor3[0], cmap='Blues', origin='lower')
    cb5 = fig.colorbar(im5, cax=cax5)
    cb5.set_label(f'{feature_dic_units[feature1]}')
    tx5 = ax5.set_title(f'Prediction: {feature_dic[feature1]}')
    min_val3 = animated_tensor3.min()
    max_val3 = animated_tensor3.max()
    cb5.set_clim(min_val3, max_val3)

    # Subplot 6
    div6 = make_axes_locatable(ax6)
    cax6 = div6.append_axes('right', '5%', '5%')

    animated_tensor4 = q_pred
    im6 = ax6.imshow(animated_tensor4[0], cmap='Greens', origin='lower')
    cb6 = fig.colorbar(im6, cax=cax6)
    cb6.set_label(f'{feature_dic_units[feature2]}')
    tx6 = ax6.set_title(f'Prediction: {feature_dic[feature2]}')
    min_val4 = animated_tensor4.min()
    max_val4 = animated_tensor4.max()
    cb6.set_clim(min_val4, max_val4)

    # Subplot 8
    div8 = make_axes_locatable(ax8)
    cax8 = div8.append_axes('right', '5%', '5%')

    animated_tensor5 = animated_tensor1 - animated_tensor3
    im8 = ax8.imshow(animated_tensor5[0], cmap='jet', origin='lower')
    cb8 = fig.colorbar(im8, cax=cax8)
    cb8.set_label(f'{feature_dic_units[feature1]}')
    tx8 = ax8.set_title(f'Diff: {feature_dic[feature1]}')
    min_val5 = animated_tensor5.min()
    max_val5 = animated_tensor5.max()
    cb8.set_clim(min_val5, max_val5)

    # Subplot 9
    div9 = make_axes_locatable(ax9)
    cax9 = div9.append_axes('right', '5%', '5%')

    animated_tensor6 = animated_tensor2 - animated_tensor4
    im9 = ax9.imshow(animated_tensor6[0], cmap='jet', origin='lower')
    cb9 = fig.colorbar(im9, cax=cax9)
    cb9.set_label(f'{feature_dic_units[feature2]}')
    tx9 = ax9.set_title(f'Diff: {feature_dic[feature2]}')
    min_val6 = animated_tensor6.min()
    max_val6 = animated_tensor6.max()
    cb9.set_clim(min_val6, max_val6)

    fig.suptitle(title_anim + f' for sample: {sample}', fontsize=16)

    def animate(i):
        # Subplot 2
        arr1 = animated_tensor1[i]
        im2.set_data(arr1)
        tx2.set_text(f'Hour {i}: {feature_dic[feature1]}')

        # Subplot 3
        arr2 = animated_tensor2[i]
        im3.set_data(arr2)
        tx3.set_text(f'Hour {i}: {feature_dic[feature2]}')

        # Subplot 5
        arr3 = animated_tensor3[i]
        im5.set_data(arr3)
        tx5.set_text(f'Prediction: {feature_dic[feature1]}')

        # Subplot 6
        arr4 = animated_tensor4[i]
        im6.set_data(arr4)
        tx6.set_text(f'Prediction: {feature_dic[feature2]}')

        # Subplot 8
        arr5 = animated_tensor5[i]
        im8.set_data(arr5)
        tx8.set_text(f'Diff: {feature_dic[feature1]}')

        # Subplot 9
        arr6 = animated_tensor6[i]
        im9.set_data(arr6)
        tx9.set_text(f'Diff: {feature_dic[feature2]}')

    # Set up the animation
    ani = animation.FuncAnimation(fig, animate, frames=water_depth.shape[0])

    # Display the animation
    plt.show()

    # Save the animation if specified
    if save:
        ani.save('post_processing/' + title_anim + '_' + model_who + '_' +
                 str(sample) + '.gif', writer='Pillow', fps=5)

    return None

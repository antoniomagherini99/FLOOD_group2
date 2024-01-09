import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

def plot_animation(sample, dataset, title_anim, save = False):
    '''

    Parameters
    ----------
    sample : int
        Choose a sample to animate.
    dataset : torch.utils.data.Dataset
        dataset of train_val or test, in future, two datasets should be parameters.
    title_anim : str
        title of the animation.
    save : bool
        The default is False, if true will save the animation in the
        post_processing folder with the title in the format 'title_anim + sample'

    Returns
    -------
    None.

    '''
    inputs = dataset[sample][0][0]
    targets = dataset[sample][1]
    
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

    # Subplot 1
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', '5%', '5%')

    static_tensor = inputs[0]  # index 0 refers to elevation
    im1 = ax1.imshow(static_tensor, origin='lower')
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label(r'$m$')
    ax1.set_title('Elevation')

    # Subplot 2
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes('right', '5%', '5%')

    animated_tensor1 = targets[:, feature1]
    im2 = ax2.imshow(animated_tensor1[0], origin='lower')
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label(f'{feature_dic_units[feature1]}')
    tx2 = ax2.set_title(f'Frame 0: {feature_dic[feature1]}')

    # Subplot 3
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes('right', '5%', '5%')

    animated_tensor2 = targets[:, feature2]
    im3 = ax3.imshow(animated_tensor2[0], origin='lower')
    cb3 = fig.colorbar(im3, cax=cax3)
    cb3.set_label(f'{feature_dic_units[feature2]}')
    tx3 = ax3.set_title(f'Frame 0: {feature_dic[feature2]}')

    fig.suptitle(title_anim + f' targets for sample: {sample}', fontsize=16)

    def animate(i):
        # Subplot 2
        arr1 = animated_tensor1[i]
        max_val1 = arr1.max()
        min_val1 = arr1.min()
        im2.set_data(arr1)
        im2.set_clim(min_val1, max_val1)
        tx2.set_text(f'Frame {i}: {feature_dic[feature1]}')

        # Subplot 3
        arr2 = animated_tensor2[i]
        max_val2 = arr2.max()
        min_val2 = arr2.min()
        im3.set_data(arr2)
        im3.set_clim(min_val2, max_val2)
        tx3.set_text(f'Frame {i}: {feature_dic[feature2]}')

    ani = animation.FuncAnimation(fig, animate, frames=targets.shape[0])

    plt.show()
    if save == True:
        ani.save('post_processing/'+ title_anim + str(sample) + '.gif', writer='Pillow', fps=20)
    return None
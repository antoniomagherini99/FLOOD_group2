# file for storing functions used for plotting results
# 1st version - Antonio

def plot_dataset_tensors(tensors, file_number, titles):
    """
    Plots a series of data tensors.

    Args:
    tensors (list of torch.Tensor): List of tensors to be plotted.
    file_number (str): Identifier of the file corresponding to the data.
    titles (list of str): Titles for each subplot corresponding to the tensors.
    """
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors, figsize=(12, 5))

    # Adjust for a single tensor case
    if num_tensors == 1:
        axes = [axes]

    for ax, tensor, title in zip(axes, tensors, titles):
        cmap = 'terrain' if title.startswith('Elevation') else 'coolwarm'
        im = ax.imshow(tensor, cmap=cmap, origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_title(title)

    plt.suptitle(f'Training Dataset Number {file_number}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return None


def resize_image(image, target_shape):
    """
    Resize the image to the target shape using PIL.
    """
    image_pil = Image.fromarray(image)
    resized_image = image_pil.resize((target_shape[1], target_shape[0]), Image.ANTIALIAS)
    return np.array(resized_image)

movie_path = os.path.join(output_dir, 'WD_Training_Dataset_Movie.mp4')
first_image = imageio.imread(filenames[1])
image_shape = first_image.shape

with imageio.get_writer(movie_path, fps=3) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        if image.shape != image_shape:
            print(f"Resizing image {filename} from {image.shape} to {image_shape}")
            image = resize_image(image, image_shape)
        writer.append_data(image)
    
    # Optionally, remove the individual image files after creating the movie
    # for filename in filenames:
    #     os.remove(filename)

    return None
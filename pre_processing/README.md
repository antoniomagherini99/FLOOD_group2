This folder contains several .py files used in the pre-processing step of both the CNN and ConvLSTM model.

# encode_decode_csv: Antonio and Lucas
This script contains functions used for encoding and decoding the datasets to load the datasets faster. We need to encode as load datasets
does not use np.loadtxt, so the runtime is painfully slow. Encode stores csvs of flattened arrays which can be easily reshaped when decoded.

# load_datasets.py: David, Apostolos, Lucas and Antonio
This file contains functions used for loading datasets (used in ConvLSTM model).

# normalization.py: Antonio and Lucas
This file contains functions used for normalizing datasets.

# process_data.py: David and Apostolos
This file contains functions for processing the elevation data and water depth data (used in CNN model initially). 

# augmentation.py: Antonio
This file contains functions used for data augmentation. The implemented data augmentation methods include rotation and horizontal flipping.
The current iteration augmentes six new datasets.
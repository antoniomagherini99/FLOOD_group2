This folder contains the notebooks, .py files and other documents related to the ConvLSTM model used in the FLOOD project.
Model will be based on model found on git: https://github.com/ndrplz/ConvLSTM_pytorch/tree/master 

# Encode_to_csv notebook
This notebook is used to encode inputs and targets datasets into .csv files for reducing the loading time. Unless something is changed in load_datasets.py or encode_into_csv.py, this notebook does not have to be re run.

# conv_lstm notebook
This notebook is used for training the ConvLSTM model

# test_augmentation
This notebook is used to test data augmentation applied to increase the size of a training dataset. The implemented data augmentation methods include rotation and horizontal flipping

# train_eval
This .py file defines functions for training and evaluation specific to a ConvLSTM model
This folder contains the notebooks, .py files and other documents related to the ConvLSTM model used in the FLOOD project.
Model will be based on model found on git: https://github.com/ndrplz/ConvLSTM_pytorch/tree/master 
Lucas and Antonio

# Encode_to_csv notebook
This notebook is used to encode inputs and targets datasets into .csv files for reducing the loading time. 
Unless something is changed in load_datasets.py or encode_into_csv.py, this notebook does not have to be re run.

# single_step_conv_lstm notebook
This notebook is used for training the Single Step ConvLSTM model. Data augmentation is applied. The implemented data augmentation methods include rotation and horizontal flipping.

# multi_step_conv_lstm
This notebook is used for training the Multi Step ConvLSTM model. Data augmentation is applied. The implemented data augmentation methods include rotation and horizontal flipping.
Some models that can be shown in the trained notebook may be overwritten, therefore this notebook only shows the last trained model.

# train_eval
This .py file defines functions for training and validating both ConvLSTM models. It could include other models but it would require minor changes. The <code>obtain_predictions</code> function is used for all the automated post processing functions.

# model_paths
Trained models can be found in this folder

# scalers
Since augmentation was different between models, their corresponding scalers can be found here.

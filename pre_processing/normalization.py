# .py file containing the functions used for normalizing datasets for DSAIE FLOOD project.

import torch
import copy

from sklearn.preprocessing import *

from pre_processing.load_datasets import count_pixels 

def scaler(train_dataset, scaler_type = MinMaxScaler()):
    '''
    Scaler function for the train dataset.

    Inputs: train_datset = training dataset including inputs and targets
            scaler_type = scaler used for normalization, default = 'MinMaxScaler()'
    
    Outputs: scaler_x, scaler_y = scalers after partial fit with train dataset for inputs and targets, respectively 
    '''

    # required to avoid that scaler_y = scaler_x
    scaler_x =  copy.deepcopy(scaler_type) 
    scaler_y =  copy.deepcopy(scaler_type)

    # get features of inputs and targets 
    in_fet = train_dataset[0][0].shape[1]
    tar_fet = train_dataset[0][1].shape[1]

    # get scalers
    for idx in range(len(train_dataset)):
        scaler_x.partial_fit(train_dataset[idx][0].reshape(in_fet, -1).T.cpu())
        permute_tar = train_dataset[idx][1].permute(1, 0, 2, 3) # place features first to help with reshaping in next line
        scaler_y.partial_fit(permute_tar.reshape(tar_fet, -1).T.cpu())
    
    return scaler_x, scaler_y

def normalize_dataset(dataset, scaler_x, scaler_y, train_val_test):
    '''
    Function for normalizing every dataset. 

    Inputs: dataset = dataset to be normalized, contains inputs and targets
            scaler_x, scaler_y = scalers for inputs and targets (water depth and discharge), created 
                                 with the scaler function
            train_val_test: key for specifying what we are using the model for
                            'train_val' = train and validate the model
                            'test1' = test the model with dataset 1
                            'test2' = test the model with dataset 2
                            'test3' = test the model with dataset 3

    Outputs: normalized_dataset = dataset after normalization 
    '''

    # get length of simulations, pixels and inputs and targets features
    len_time = dataset[0][1].shape[0]
    pixels = count_pixels(train_val_test)
    input_features = dataset[0][0].shape[1]
    tar_features = dataset[0][1].shape[1]
    
    # initialize list for looping
    normalized_dataset = []

    # normalize dataset by looping over all elements
    for idx in range(len(dataset)):
        norm_x = torch.FloatTensor(scaler_x.transform(
            dataset[idx][0][0].reshape(input_features, -1).T).T.reshape((1, input_features, pixels, pixels))) # 1 time step
        
        targets = dataset[idx][1]
        feature_first_tar = targets.permute(1, 0, 2, 3) # place features first
        norm_y = torch.FloatTensor(scaler_y.transform(
            feature_first_tar.reshape(tar_features, -1).T).T.reshape(tar_features, len_time, pixels, pixels))
        
        normalized_dataset.append((norm_x, norm_y.permute(1, 0, 2, 3))) # place time steps first
    
    return normalized_dataset

def denormalize_dataset(inputs, outputs, train_val_test, scaler_x, scaler_y):
    '''
    Function for denormalizing the inputs and targets/outputs for a single sample. 

    Inputs: dataset = dataset to be normalized
            train_val_test: key for specifying what we are using the model for
                            'train_val' = train and validate the model
                            'test1' = test the model with dataset 1
                            'test2' = test the model with dataset 2
                            'test3' = test the model with dataset 3
            scaler_x, scaler_y = scalers for inputs and targets (water depth and discharge), created 
                                            with the scaler function

    Outputs: denormalized elevation, water_depth and discharge 
    '''
    
    # get length of simulations, pixels and inputs and targets features
    len_time = outputs.shape[0]    
    pixels = count_pixels(train_val_test)
    input_features = inputs.shape[1]
    tar_features = outputs.shape[1]

    # reorganize dimensions of dataset
    feature_first_output = outputs.permute(1, 0, 2, 3)
    
    # denormalize inputs and targets 
    denorm_in = scaler_x.inverse_transform(inputs.reshape(input_features, -1).T.cpu()).T.reshape(input_features, pixels, pixels)
    elevation = denorm_in[0]

    denorm_out = scaler_y.inverse_transform(feature_first_output.reshape(tar_features, -1).T.cpu()).T.reshape(tar_features, len_time, pixels, pixels)
    water_depth = denorm_out[0]
    discharge = denorm_out[1]

    return elevation, water_depth, discharge
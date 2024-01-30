import torch
from models.ConvLSTM_model.train_eval import obtain_predictions

def threshold_function(tensor, thresholds, scaler_y):
     '''
     Return a binary tensor which has ones at non zero elements
     and zeros at elements below the predefined threshold for each feature.
     
     Parameters
     ----------
     tensor: torch.tensor
         Normalized tensor that has shape: 
         (time_steps x num_features x pixel x pixel). Function can be used
         by targets or model outputs.
     thresholds: torch.tensor
         Denormalized thresholds for each feature. Expects a tensor that
         has shape: (1 x num_features).
    scaler_y : instance of normalizer
        Used on the targets/outputs.
    
    Returns
    -------
    binary_tensor: torch.tensor
        Tensor of shape (time_steps x num_features x pixel x pixel) which
        contains 0 at indicies where the water depth or discharge is below
        their respective threshold (indicates a negative) and 1 at indicies 
        where the values are above the threshold (indicates a positive).
        Postivie means flooded.
    
     '''
     # tensors are expected to be normalized
     norm_thresholds = scaler_y.transform(thresholds)
     
     feature_list = []
     for feature in range(tensor.shape[1]):
         feature_threshold = norm_thresholds[0, feature]
         feature_tensor = tensor[:, feature]
     
         binary_feature = torch.where(
             feature_tensor > feature_threshold,
             torch.tensor(1), torch.tensor(0)
             ).unsqueeze(1) # recreate index 1 to concat later
         
         feature_list.append(binary_feature)
     binary_tensor = torch.cat(feature_list, dim = 1)
     return binary_tensor
 
def binary_accuracy(pred_binary, target_binary):
    '''
    Compute accuracy based on predictions and targets

    Parameters
    ----------
    pred_binary : torch.tensor
        Tensor of any shape. Binary, with 1 at positives and 0 at negatives.
        Represents the prediction of the model.
    target_binary : torch.tensor
        Tensor of any shape, but must match the shape of pred_binary. Binary,
        with 1 at positives and 0 at negatives. Represents the targets.

    Returns
    -------
    Accuracy : float
        Quotient of number of correct predictions (TP + TN)
        and number of total elements (TP + FN + FP + TN).
        Value should range between 0 and 1 where a 1 indicates predicted
        values match actual values.
        
    '''
    correct_predictions = torch.sum(pred_binary == target_binary)
    total_elements = pred_binary.numel()
    accuracy = correct_predictions / total_elements
    return accuracy

def binary_recall(pred_binary, target_binary):
    '''
    Compute recall based on predictions and targets

    Parameters
    ----------
    pred_binary : torch.tensor
        Tensor of any shape. Binary, with 1 at positives and 0 at negatives.
        Represents the prediction of the model.
    target_binary : torch.tensor
        Tensor of any shape, but must match the shape of pred_binary. Binary,
        with 1 at positives and 0 at negatives. Represents the targets.

    Returns
    -------
    recall : float
        Quotient of number of true postivies (TP)
        and number of actual positives (TP + FN).
        Value should range between 0 and 1 where a 1 indicates predicted
        positives match actual postives.
        If actual positives is equal to 0, recall is 0.

    '''
    true_positives = torch.sum(pred_binary * target_binary)
    actual_positives = torch.sum(target_binary)
    if actual_positives > 0:
        recall = true_positives / actual_positives
    else:
        recall = 0.0 # float
    return recall

def binary_precision(pred_binary, target_binary):
    '''
    Compute precision based on predictions and targets

    Parameters
    ----------
    pred_binary : torch.tensor
        Tensor of any shape. Binary, with 1 at positives and 0 at negatives.
        Represents the prediction of the model.
    target_binary : torch.tensor
        Tensor of any shape, but must match the shape of pred_binary. Binary,
        with 1 at positives and 0 at negatives. Represents the targets.

    Returns
    -------
    Precision : float
        Quotient of number of true positives (TP)
        and number of predicted positives (TP + FP).
        Value should range between 0 and 1 where a 1 indicates predicted
        positives match true postives.
        If predicted positives is equal to 0, precision is 0.
        
    '''
    true_positives = torch.sum(pred_binary * target_binary)
    predicted_positives = torch.sum(pred_binary)
    if predicted_positives > 0:
        precision = true_positives / predicted_positives
    else:
        precision = 0.0 # float
    return precision

def binary_f1_score(pred_binary, target_binary):    
    '''
    Compute the f1 score based on predictions and targets

    Parameters
    ----------
    pred_binary : torch.tensor
        Tensor of any shape. Binary, with 1 at positives and 0 at negatives.
        Represents the prediction of the model.
    target_binary : torch.tensor
        Tensor of any shape, but must match the shape of pred_binary. Binary,
        with 1 at positives and 0 at negatives. Represents the targets.

    Returns
    -------
    f1 : float
        Harmonic mean of precision and recall to give a better overview of the
        capacity of the model. Value ranges between 0 and 1, where 1 indicates
        the model is perfect at predicting flooded cells.
        If the sum of precision and recall is 0, f1 score is 0.

    '''
    precision = binary_precision(pred_binary, target_binary)
    recall = binary_recall(pred_binary, target_binary)
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1

def confusion_mat(dataset, model, scaler_y, device,
                  thresholds = torch.tensor([0.1, 0]).reshape(1, -1),
                  sample = False, sample_num = 0):
    '''
    Compute recall, accuracy and f1 score for a dataset (or a sample
    of a dataset if sample is True).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Normalized dataset of train_val or test (1, 2, or 3)
    model : class of model
        Model used to create predictions
    scaler_y : instance of normalizer
        Used on the targets/outputs
    device : str
        Device on which to perform the computations
    thresholds : torch.tensor, optional
        Denormalized thresholds for each feature. Expects a tensor that
        has shape: (1 x num_features).
        The default is torch.tensor([0.1, 0]).reshape(1, -1).
    sample : bool, optional
        Tells the function whether to compute metrics on entire dataset or
        just a sample. The default is False.
    sample_num : int, optional
        Only used if sample is True. Indicates which sample the metrics
        will be calculated for. The default is 0.

    Returns
    -------
    Shape of outputs depends on whether sample is set to True or False.
    If sample is True: shape is (features x time_steps)
    if sample is false: shape is (len_dataset) (only computes metrics for water
    depth)
    
    recall_arr : torch.tensor
        If sample is True: computes recall for all features and time steps of
        a sample
        Uf sample is False: computes recall for water depth for all samples
    accuracy_arr : torch.tensor
        If sample is True: computes accuracy for all features and time steps of
        a sample
        If sample is False: computes accuracy for water depth for all samples
    f1_arr : torch.tensor
        If sample is True: computes f1 for all features and time steps of
        a sample
        If sample is False: computes f1 for water depth for all samples

    '''
    # only needed if sample == False
    metrics_tensor = torch.zeros((3, len(dataset))) # recall, accuracy, f1
    
    for samples in range(len(dataset)):
        if sample == True:
            samples = sample_num
        else:
            None
        input = dataset[samples][0]
        target = dataset[samples][1]
        time_steps = target.shape[0]
        
        preds = obtain_predictions(model, input, device, time_steps)
        
        target_binary = threshold_function(target, thresholds, scaler_y)
        pred_binary = threshold_function(preds, thresholds, scaler_y)
        
        fet = target.shape[1]
        time_steps = target.shape[0]
        
        recall_arr = torch.zeros((fet, time_steps))
        accuracy_arr = torch.zeros((fet, time_steps))
        f1_arr = torch.zeros((fet, time_steps))
        for feature in range(fet):
            for step in range(time_steps):
                recall_arr[feature][step] = binary_recall(pred_binary[step][feature], target_binary[step][feature])
                accuracy_arr[feature][step] = binary_accuracy(pred_binary[step][feature], target_binary[step][feature])
                f1_arr[feature][step] = binary_f1_score(pred_binary[step][feature], target_binary[step][feature])
        if sample == False:
            metrics_tensor[0, samples] = torch.mean(recall_arr[0]) # macro average of water depth only
            metrics_tensor[1, samples] = torch.mean(accuracy_arr[0])
            metrics_tensor[2, samples] = torch.mean(f1_arr[0])
        elif sample == True:
            break
    if sample == False:
        recall_arr = metrics_tensor[0]
        accuracy_arr = metrics_tensor[1]
        f1_arr = metrics_tensor[2]
    else:
        None
    return recall_arr, accuracy_arr, f1_arr

import torch
from models.ConvLSTM_model.train_eval import obtain_predictions

def threshold_function(tensor):
     '''
     Use a tensor to return a binary tensor which has ones at non zero elements
     and zeros at zero elements
     '''
     ## Too lazy to implement scaler_y in all functions.
     ## Needed if we want to include a threshold which is non zero.
     return torch.where(tensor > 0, torch.tensor(1), torch.tensor(0))

def binary_recall(pred_binary, target_binary):
    true_positives = torch.sum(pred_binary * target_binary)
    actual_positives = torch.sum(target_binary)
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def binary_accuracy(pred_binary, target_binary):
    correct_predictions = torch.sum(pred_binary == target_binary)
    total_elements = pred_binary.numel()
    return correct_predictions / total_elements

def binary_f1_score(pred_binary, target_binary):
    true_positives = torch.sum(pred_binary * target_binary)
    predicted_positives = torch.sum(pred_binary)
    actual_positives = torch.sum(target_binary)
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def confusion_mat(dataset, model, device, sample = False, sample_num = 0):
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
        
        target_binary = threshold_function(target)
        pred_binary = threshold_function(preds)
        
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

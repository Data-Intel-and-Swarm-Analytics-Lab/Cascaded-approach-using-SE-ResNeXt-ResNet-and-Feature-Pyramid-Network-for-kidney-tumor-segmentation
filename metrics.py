import torch

# def dice_coefficient(y_true, y_pred, smooth=1.0):
#     intersection = torch.sum(y_true * y_pred)
#     union = torch.sum(y_true) + torch.sum(y_pred)
#     dice = (2.0 * intersection + smooth) / (union + smooth)
#     return dice

def dice_coefficient(y_true, y_pred):
    y_true = torch.tensor(y_true, dtype=torch.float32)  # Convert to PyTorch tensor
    y_pred = torch.tensor(y_pred, dtype=torch.float32)  # Convert to PyTorch tensor

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    return dice

# def dice_coefficient(y_true, y_pred, smooth=1.0):
#     y_true = torch.tensor(y_true, dtype=torch.float32)  # Convert to PyTorch tensor
#     y_pred = torch.tensor(y_pred, dtype=torch.float32)  # Convert to PyTorch tensor

#     intersection = torch.sum(y_true * y_pred)
#     union = torch.sum(y_true) + torch.sum(y_pred)
#     dice = (2.0 * intersection + smooth) / (union + smooth)
#     return dice

# def intersection_over_union(y_true, y_pred, smooth=1.0):      #The smooth parameter is added to avoid division by zero.
#     intersection = torch.sum(y_true * y_pred)
#     union = torch.sum(y_true) + torch.sum(y_pred) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     return iou

def jaccard_index(y_true, y_pred):
    y_true = torch.tensor(y_true, dtype=torch.float32)  # Convert to PyTorch tensor
    y_pred = torch.tensor(y_pred, dtype=torch.float32)  # Convert to PyTorch tensor

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    jaccard = (intersection + 1e-5) / (union + 1e-5)
    return jaccard

def get_precision(y_true, y_pred, smooth=1.0):
    y_true = torch.tensor(y_true, dtype=torch.float32)  # Convert to PyTorch tensor
    y_pred = torch.tensor(y_pred, dtype=torch.float32)  # Convert to PyTorch tensor
    true_positives = torch.sum(y_true * y_pred)
    false_positives = torch.sum(y_pred) - true_positives
    precision = (true_positives + smooth) / (true_positives + false_positives + smooth)
    return precision

def get_recall(y_true, y_pred, smooth=1.0):
    y_true = torch.tensor(y_true, dtype=torch.float32)  # Convert to PyTorch tensor
    y_pred = torch.tensor(y_pred, dtype=torch.float32)  # Convert to PyTorch tensor
    true_positives = torch.sum(y_true * y_pred)
    false_negatives = torch.sum(y_true) - true_positives
    recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)
    return recall


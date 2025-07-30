from scipy.spatial.distance import directed_hausdorff
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
import torch.nn as nn


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def Hausdorff_distance(pred, label):
    pred = np.array(pred.cpu())
    pred_points = np.argwhere(pred == 1)
    label = np.array(label.cpu())
    label_points = np.argwhere(label == 1)
    HD95, _, _ = directed_hausdorff(pred_points, label_points)
    return HD95


def ASD(pred, label, voxel_spacing=1):
    pred = np.array(pred.cpu())
    pred_points = np.argwhere(pred == 1)
    label = np.array(label.cpu())
    label_points = np.argwhere(label == 1)
    distances = cdist(pred_points, label_points) * voxel_spacing
    if len(distances) > 0:
        min_distances = np.min(distances, axis=1)
    else:
        min_distances = np.nan
    asd = np.mean(min_distances)
    return asd


def miou_function(y_true, y_pred, num_classes):
    y_true, y_pred = y_true.cpu(), y_pred.cpu()
    iou = []
    for c in range(num_classes):
        true_class = y_true == c
        pred_class = y_pred == c
        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()
        if union == 0:
            iou.append(float("nan"))
        else:
            iou.append(intersection / union)
    return np.nanmean(iou)


def iou_function(y_true, y_pred, classes_index):
    smooth = 1e-10
    y_true, y_pred = y_true.cpu(), y_pred.cpu()
    true_class = y_true == classes_index
    pred_class = y_pred == classes_index
    intersection = np.logical_and(true_class, pred_class).sum()
    union = np.logical_or(true_class, pred_class).sum()
    return intersection / (union + smooth)


def specificity(y_true, y_pred):
    TN = torch.sum((y_pred == 0) & (y_true == 0)).item()
    FP = torch.sum((y_pred == 1) & (y_true == 0)).item()

    if TN + FP == 0:
        return 1.0
    return TN / (TN + FP)


def sensitivity(y_true, y_pred):
    TP = torch.sum((y_pred == 1) & (y_true == 1)).item()
    FN = torch.sum((y_pred == 0) & (y_true == 1)).item()

    if TP + FN == 0:
        return 1.0
    return TP / (TP + FN)


class Dice_and_ce_loss(nn.Module):
    def __init__(self):
        super(Dice_and_ce_loss, self).__init__()

    def forward(self, pred, label):
        outputs_soft = F.softmax(pred, dim=1)
        dice = dice_loss(outputs_soft[:, 1, :, :], label == 1)
        CE_loss = F.cross_entropy(pred, label, ignore_index=255)  # logits [b c h w] label [b h w]
        loss = dice + CE_loss
        return dice, CE_loss, loss

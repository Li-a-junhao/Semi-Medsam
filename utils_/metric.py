from utils_.metrics import miou_function, iou_function, dice_loss, specificity, sensitivity


def calculate_metrics(pred, label):
    pred[label == 255] = 255
    dice_0 = 1 - dice_loss(pred == 0, label == 0)
    dice_1 = 1 - dice_loss(pred == 1, label == 1)
    dice_avg = (dice_0 + dice_1) / 2

    iou_0 = iou_function(y_true=label, y_pred=pred, classes_index=0)
    iou_1 = iou_function(y_true=label, y_pred=pred, classes_index=1)
    miou = miou_function(y_true=label, y_pred=pred, num_classes=2)

    sp = specificity(label, pred)
    se = sensitivity(label, pred)
    return [dice_avg, dice_0, dice_1, miou, iou_0, iou_1, sp, se]


def sum_metrics(metric_total, metric):
    for i in range(len(metric_total)):
        metric_total[i] = metric_total[i] + metric[i]
    return metric_total
